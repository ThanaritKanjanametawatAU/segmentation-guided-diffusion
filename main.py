import os
from argparse import ArgumentParser

# torch imports
import torch
from torch import nn
from torchvision import transforms
import torch.nn.functional as F
import numpy as np

# HF imports
import diffusers
from diffusers.optimization import get_cosine_schedule_with_warmup
import datasets
from datasets import load_dataset

# torchvision imports
import torchvision

# custom imports
from training import TrainingConfig, train_loop
from eval import evaluate_generation, evaluate_sample_many

def main(
    mode,
    img_size,
    num_img_channels,
    dataset,
    img_dir,
    seg_dir,
    model_type,
    segmentation_guided,
    segmentation_channel_mode,
    num_segmentation_classes,
    train_batch_size,
    eval_batch_size,
    num_epochs,
    resume_epoch=None,
    use_ablated_segmentations=False,
    eval_shuffle_dataloader=True,
    save_image_epochs=20,
    save_model_epochs=30,

    # arguments only used in eval
    eval_mask_removal=False,
    eval_blank_mask=False,
    eval_sample_size=1000
):
    


    #GPUs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('running on {}'.format(device))




    # load config
    dataset_formatted = dataset.replace('/', '_')
    output_dir = '{}-{}-{}'.format(model_type.lower(), dataset_formatted, img_size)  # the model namy locally and on the HF Hub
    if segmentation_guided:
        output_dir += "-segguided"
        assert seg_dir is not None, "must provide segmentation directory for segmentation guided training/sampling"

    if use_ablated_segmentations or eval_mask_removal or eval_blank_mask:
        output_dir += "-ablated"

    print("output dir: {}".format(output_dir))

    if mode == "train":
        evalset_name = "val"
        # Modify this check to allow HF datasets
        if dataset.startswith("Thanarit/") or dataset.startswith("huggingface/"):
            # Using Hugging Face dataset
            pass
        else:
            # Using local directory
            assert img_dir is not None, "must provide image directory for training when not using Hugging Face dataset"
    elif "eval" in mode:
        evalset_name = "test"

    print("using evaluation set: {}".format(evalset_name))

    config = TrainingConfig(
        image_size = img_size,
        dataset = dataset,
        segmentation_guided = segmentation_guided,
        segmentation_channel_mode = segmentation_channel_mode,
        num_segmentation_classes = num_segmentation_classes,
        train_batch_size = train_batch_size,
        eval_batch_size = eval_batch_size,
        num_epochs = num_epochs,
        output_dir = output_dir,
        save_image_epochs=save_image_epochs,
        save_model_epochs=save_model_epochs,
        model_type=model_type,
        resume_epoch=resume_epoch,
        use_ablated_segmentations=use_ablated_segmentations
    )

    load_images_as_np_arrays = False
    if num_img_channels not in [1, 3]:
        load_images_as_np_arrays = True
        print("image channels not 1 or 3, attempting to load images as np arrays...")

    if config.segmentation_guided:
        seg_types = os.listdir(seg_dir)
        seg_paths_train = {} 
        seg_paths_eval = {}

        # train set
        if img_dir is not None: 
            # make sure the images are matched to the segmentation masks
            img_dir_train = os.path.join(img_dir, "train")
            img_paths_train = [os.path.join(img_dir_train, f) for f in os.listdir(img_dir_train)]
            for seg_type in seg_types:
                seg_paths_train[seg_type] = [os.path.join(seg_dir, seg_type, "train", f) for f in os.listdir(img_dir_train)]
        else:
            for seg_type in seg_types:
                seg_paths_train[seg_type] = [os.path.join(seg_dir, seg_type, "train", f) for f in os.listdir(os.path.join(seg_dir, seg_type, "train"))]

        # eval set
        if img_dir is not None: 
            img_dir_eval = os.path.join(img_dir, evalset_name)
            img_paths_eval = [os.path.join(img_dir_eval, f) for f in os.listdir(img_dir_eval)]
            for seg_type in seg_types:
                seg_paths_eval[seg_type] = [os.path.join(seg_dir, seg_type, evalset_name, f) for f in os.listdir(img_dir_eval)]
        else:
            for seg_type in seg_types:
                seg_paths_eval[seg_type] = [os.path.join(seg_dir, seg_type, evalset_name, f) for f in os.listdir(os.path.join(seg_dir, seg_type, evalset_name))]

        if img_dir is not None:
            dset_dict_train = {
                    **{"image": img_paths_train},
                    **{"seg_{}".format(seg_type): seg_paths_train[seg_type] for seg_type in seg_types}
                }
            
            dset_dict_eval = {
                    **{"image": img_paths_eval},
                    **{"seg_{}".format(seg_type): seg_paths_eval[seg_type] for seg_type in seg_types}
            }
        else:
            dset_dict_train = {
                    **{"seg_{}".format(seg_type): seg_paths_train[seg_type] for seg_type in seg_types}
                }
            
            dset_dict_eval = {
                    **{"seg_{}".format(seg_type): seg_paths_eval[seg_type] for seg_type in seg_types}
            }


        if img_dir is not None:
            # add image filenames to dataset
            dset_dict_train["image_filename"] = [os.path.basename(f) for f in dset_dict_train["image"]]
            dset_dict_eval["image_filename"] = [os.path.basename(f) for f in dset_dict_eval["image"]]
        else:
            # use segmentation filenames as image filenames
            dset_dict_train["image_filename"] = [os.path.basename(f) for f in dset_dict_train["seg_{}".format(seg_types[0])]]
            dset_dict_eval["image_filename"] = [os.path.basename(f) for f in dset_dict_eval["seg_{}".format(seg_types[0])]]

        dataset_train = datasets.Dataset.from_dict(dset_dict_train)
        dataset_eval = datasets.Dataset.from_dict(dset_dict_eval)

        # load the images
        if not load_images_as_np_arrays and img_dir is not None:
            dataset_train = dataset_train.cast_column("image", datasets.Image())
            dataset_eval = dataset_eval.cast_column("image", datasets.Image())

        for seg_type in seg_types:
            dataset_train = dataset_train.cast_column("seg_{}".format(seg_type), datasets.Image())

        for seg_type in seg_types:
            dataset_eval = dataset_eval.cast_column("seg_{}".format(seg_type), datasets.Image())

    else:
        if img_dir is not None:
            img_dir_train = os.path.join(img_dir, "train")
            img_paths_train = [os.path.join(img_dir_train, f) for f in os.listdir(img_dir_train)]

            img_dir_eval = os.path.join(img_dir, evalset_name)
            img_paths_eval = [os.path.join(img_dir_eval, f) for f in os.listdir(img_dir_eval)]

            dset_dict_train = {
                    **{"image": img_paths_train}
                }

            dset_dict_eval = {
                    **{"image": img_paths_eval}
                }

            # add image filenames to dataset
            dset_dict_train["image_filename"] = [os.path.basename(f) for f in dset_dict_train["image"]]
            dset_dict_eval["image_filename"] = [os.path.basename(f) for f in dset_dict_eval["image"]]

            dataset_train = datasets.Dataset.from_dict(dset_dict_train)
            dataset_eval = datasets.Dataset.from_dict(dset_dict_eval)

            # load the images
            if not load_images_as_np_arrays:
                dataset_train = dataset_train.cast_column("image", datasets.Image())
                dataset_eval = dataset_eval.cast_column("image", datasets.Image())

    # training set preprocessing
    if not load_images_as_np_arrays:
        preprocess = transforms.Compose(
            [
                transforms.Resize((config.image_size, config.image_size)),
                # transforms.RandomHorizontalFlip(), # flipping wouldn't result in realistic images
                transforms.ToTensor(),
                transforms.Normalize(
                    num_img_channels * [0.5], 
                    num_img_channels * [0.5]),
            ]
        )
    else:
        # resizing will be done in the transform function
        preprocess = transforms.Compose(
            [
                transforms.Normalize(
                    num_img_channels * [0.5], 
                    num_img_channels * [0.5]),
            ]
        )

    if num_img_channels == 1:
        PIL_image_type = "L"
    elif num_img_channels == 3:
        PIL_image_type = "RGB"
    else:
        PIL_image_type = None

    if config.segmentation_guided:
        preprocess_segmentation = transforms.Compose(
        [
            transforms.Resize((config.image_size, config.image_size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ]
        )

        def transform(examples):
            if img_dir is not None:
                if not load_images_as_np_arrays:
                    images = [preprocess(image.convert(PIL_image_type)) for image in examples["image"]]
                else:
                    # load np array as torch tensor, resize, then normalize
                    images = [
                        preprocess(F.interpolate(torch.tensor(np.load(image)).unsqueeze(0).float(), size=(config.image_size, config.image_size)).squeeze()) for image in examples["image"]
                        ]

            images_filenames = examples["image_filename"]

            segs = {}
            for seg_type in seg_types:
                segs["seg_{}".format(seg_type)] = [preprocess_segmentation(image.convert("L")) for image in examples["seg_{}".format(seg_type)]]
            # return {"images": images, "seg_breast": seg_breast, "seg_dv": seg_dv}
            if img_dir is not None:
                return {**{"images": images}, **segs, **{"image_filenames": images_filenames}}
            else:
                return {**segs, **{"image_filenames": images_filenames}}
            
        dataset_train.set_transform(transform)
        dataset_eval.set_transform(transform)

    else:
        if img_dir is not None:
            pass

        def transform(examples):
            # Preprocessing for both master and defect images
            preprocess = transforms.Compose(
                [
            transforms.Resize((config.image_size, config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

            if not load_images_as_np_arrays:
                # Load both master (normal) and defect PCB images
                master_images = [preprocess(image.convert("RGB")) for image in examples["master_image"]]
                defect_images = [preprocess(image.convert("RGB")) for image in examples["defect_image"]]
            else:
                master_images = [
                    preprocess(F.interpolate(torch.tensor(image).unsqueeze(0).float(), 
                    size=(config.image_size, config.image_size)).squeeze()) 
                    for image in examples["master_image"]
                ]
                defect_images = [
                    preprocess(F.interpolate(torch.tensor(image).unsqueeze(0).float(),
                    size=(config.image_size, config.image_size)).squeeze())
                    for image in examples["defect_image"]
                ]
    
            # Add filenames for tracking
            images_filenames = examples["pair_id"]
    
            return {
                "master_images": master_images,
                "defect_images": defect_images,
                "image_filenames": images_filenames
            }

        if dataset.startswith("Thanarit/"):
            # Load HuggingFace dataset
            ds = load_dataset(dataset)
            dataset_train = ds["train"]
            
            # Select only first 100 examples for training
            dataset_train = dataset_train.select(range(1000))
            
            # Cast image columns 
            dataset_train = dataset_train.cast_column("master_image", datasets.Image())
            dataset_train = dataset_train.cast_column("defect_image", datasets.Image())
            
            # Set transform
            dataset_train.set_transform(transform)
            
            # Create training dataloader
            train_dataloader = torch.utils.data.DataLoader(
                dataset_train, 
                batch_size=config.train_batch_size,
                shuffle=True
            )
            
            # For eval/testing, use just one specific example
            specific_example = dataset_train.select([0])  # Select first example
            eval_dataloader = torch.utils.data.DataLoader(
                specific_example,
                batch_size=1,  # Keep batch size 1 since we're using one example 
                shuffle=False  # Don't shuffle to get same example each time
            )

    # define the model
    in_channels = num_img_channels
    if config.segmentation_guided:
        assert config.num_segmentation_classes is not None
        assert config.num_segmentation_classes > 1, "must have at least 2 segmentation classes (INCLUDING background)" 
        if config.segmentation_channel_mode == "single":
            in_channels += 1
        elif config.segmentation_channel_mode == "multi":
            in_channels = len(seg_types) + in_channels

    model = diffusers.UNet2DModel(
        sample_size=config.image_size,
        in_channels=num_img_channels * 2,  # Double channels for concatenated input
        out_channels=num_img_channels,
        layers_per_block=2,
        block_out_channels=(128, 128, 256, 256, 512, 512),
        down_block_types=( 
            "DownBlock2D",
            "DownBlock2D", 
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
            ),
        up_block_types=(
            "UpBlock2D",
            "AttnUpBlock2D", 
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D"
    ),
)

    if (mode == "train" and resume_epoch is not None) or "eval" in mode:
        if mode == "train":
            print("resuming from model at training epoch {}".format(resume_epoch))
        elif "eval" in mode:
            print("loading saved model...")
        print("Dir: " + os.path.join(config.output_dir))
        model = model.from_pretrained(os.path.join(config.output_dir, 'unet'), use_safetensors=True)

    model = nn.DataParallel(model)
    model.to(device)

    # define noise scheduler
    if model_type == "DDPM":
        noise_scheduler = diffusers.DDPMScheduler(num_train_timesteps=1000)
    elif model_type == "DDIM":
        noise_scheduler = diffusers.DDIMScheduler(num_train_timesteps=1000)

    if mode == "train":
        # training setup
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=config.lr_warmup_steps,
            num_training_steps=(len(train_dataloader) * config.num_epochs),
        )

        # train
        train_loop(
            config, 
            model, 
            noise_scheduler, 
            optimizer, 
            train_dataloader, 
            eval_dataloader, 
            lr_scheduler, 
            device=device
            )
    elif mode == "eval":
        """
        default eval behavior:
        evaluate image generation or translation (if for conditional model, either evaluate naive class conditioning but not CFG,
        or with CFG),
        possibly conditioned on masks.

        has various options.
        """
        evaluate_generation(
            config, 
            model, 
            noise_scheduler,
            eval_dataloader, 
            eval_mask_removal=eval_mask_removal,
            eval_blank_mask=eval_blank_mask,
            device=device
            )

    elif mode == "eval_many":
        """
        generate many images and save them to a directory, saved individually
        """
        evaluate_sample_many(
            eval_sample_size,
            config,
            model,
            noise_scheduler,
            eval_dataloader,
            device=device
            )

    else:
        raise ValueError("mode \"{}\" not supported.".format(mode))


if __name__ == "__main__":
    # parse args:
    parser = ArgumentParser()
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--num_img_channels', type=int, default=1)
    parser.add_argument('--dataset', type=str, default="breast_mri")
    parser.add_argument('--img_dir', type=str, default=None)
    parser.add_argument('--seg_dir', type=str, default=None)
    parser.add_argument('--model_type', type=str, default="DDPM")
    parser.add_argument('--segmentation_guided', action='store_true', help='use segmentation guided training/sampling?')
    parser.add_argument('--segmentation_channel_mode', type=str, default="single", help='single == all segmentations in one channel, multi == each segmentation in its own channel')
    parser.add_argument('--num_segmentation_classes', type=int, default=None, help='number of segmentation classes, including background')
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--eval_batch_size', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--resume_epoch', type=int, default=None, help='resume training starting at this epoch')

    # novel options
    parser.add_argument('--use_ablated_segmentations', action='store_true', help='use mask ablated training and any evaluation? sometimes randomly remove class(es) from mask during training and sampling.')

    # other options
    parser.add_argument('--eval_noshuffle_dataloader', action='store_true', help='if true, don\'t shuffle the eval dataloader')

    # args only used in eval
    parser.add_argument('--eval_mask_removal', action='store_true', help='if true, evaluate gradually removing anatomies from mask and re-sampling')
    parser.add_argument('--eval_blank_mask', action='store_true', help='if true, evaluate sampling conditioned on blank (zeros) masks')
    parser.add_argument('--eval_sample_size', type=int, default=1000, help='number of images to sample when using eval_many mode')

    parser.add_argument('--save_image_epochs', type=int, default=20, help='save sample images every N epochs')
    parser.add_argument('--save_model_epochs', type=int, default=30, help='save model checkpoint every N epochs')

    args = parser.parse_args()

    main(
        args.mode,
        args.img_size,
        args.num_img_channels,
        args.dataset,
        args.img_dir,
        args.seg_dir,
        args.model_type,
        args.segmentation_guided,
        args.segmentation_channel_mode,
        args.num_segmentation_classes,
        args.train_batch_size,
        args.eval_batch_size,
        args.num_epochs,
        args.resume_epoch,
        args.use_ablated_segmentations,
        not args.eval_noshuffle_dataloader,
        args.save_image_epochs,
        args.save_model_epochs, 

        # args only used in eval
        args.eval_mask_removal,
        args.eval_blank_mask,
        args.eval_sample_size
    )
