import gradio as gr
import torch
from PIL import Image
import numpy as np
from datasets import load_dataset
import os
from inpaintinference import PCBLatentInpaintingInference
import torchvision.transforms as transforms
from torchvision.utils import save_image
from image_gen_aux import UpscaleWithModel

class PCBDefectGenerator:
    def __init__(self, model_path="working_model/checkpoint-2999-126images-CLIP", upscaler_path="models/4xNomos8kDAT.pth"):
        self.model = PCBLatentInpaintingInference(model_path)
        self.dataset = load_dataset("Thanarit/PCB-v3")["train"]
        self.upscaler = UpscaleWithModel.from_pretrained("Phips/4xNomosUniDAT2_multijpg_ldl_sharp").to("cuda")
        
        # Create examples directory if it doesn't exist
        self.examples_dir = "example_images"
        os.makedirs(self.examples_dir, exist_ok=True)
        
        # Prepare example images
        self.examples = self.prepare_examples()

    def prepare_examples(self):
        examples = []
        pairs = [92, 6, 89, 76, 111]
        for idx in range(len(self.dataset)):
            example = self.dataset[idx]
            if int(example["pair_id"]) in pairs:
                # Save image to file
                img_path = os.path.join(self.examples_dir, f"example_{idx}.png")
                example["master_image"].save(img_path)
                examples.append([
                    img_path,  # Path to saved image
                    example["description"]  # Description
                ])
        return examples
    
    def generate_defect(self, master_image, mask_image, prompt, apply_upscale=True):
        # Handle Gradio sketch tool output format
        if isinstance(master_image, dict):
            base_image = master_image["image"]
            mask = master_image["mask"]
        else:
            base_image = master_image
            mask = mask_image["mask"] if mask_image is not None else None

        # Convert numpy arrays to PIL Images
        if isinstance(base_image, np.ndarray):
            base_image = Image.fromarray(base_image)
        if isinstance(mask, np.ndarray):
            mask = Image.fromarray(mask)

        transform_image = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        transform_mask = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ])

        master_image_transformed = transform_image(base_image.convert("RGB")).unsqueeze(0)
        mask_transformed = transform_mask(mask.convert("L")).unsqueeze(0) if mask is not None else torch.zeros_like(master_image_transformed)

        generated_image = self.model.generate(
            prompt=prompt,
            master_image=master_image_transformed,
            mask=mask_transformed,
            num_inference_steps=100
        )
        
        # Save the raw generated image
        save_image(generated_image, "generated_image.png")
        
        # Apply upscaling if requested
        if apply_upscale:
            # Convert tensor to PIL for upscaling
            pil_image = self.tensor_to_pil(generated_image)
            upscaled_image = self.upscale_image(pil_image)
            # Save upscaled image
            upscaled_image.save("upscaled_image.png")
            return upscaled_image
        
        return self.tensor_to_pil(generated_image)
    
    def upscale_image(self, image):
        upscaled = self.upscaler(image, tiling=True, tile_width=1024, tile_height=1024)
        return upscaled

    def create_interface(self):
        with gr.Blocks(css="""
            .container {max-width: 1200px; margin: auto;}
            .image-display {min-height: 512px}
            """) as interface:
            
            gr.Markdown("""
            # PCB Defect Generator
            Draw a mask on the PCB image and provide a description of the defect you want to generate.
            """)

            with gr.Row():
                with gr.Column():
                    image_input = gr.Image(
                        source="upload",
                        tool="sketch",
                        type="numpy",
                        label="Draw defect mask on PCB image",
                        shape=(512, 512), 
                        height=512
                    )
                    
                    prompt_input = gr.Textbox(
                        label="Defect Description",
                        placeholder="Describe the defect you want to generate..."
                    )
                    
                    apply_upscaling = gr.Checkbox(label="Apply 4x Upscaling", value=True)
                    generate_btn = gr.Button("Generate Defect", variant="primary")

                with gr.Column():
                    output_image = gr.Image(
                        label="Generated Defective PCB",
                        height=512,
                        width=512,
                        type="pil",
                        show_download_button=True
                    )

            # Example selector
            examples = self.prepare_examples()
            gr.Examples(
                examples=examples,
                inputs=[image_input, prompt_input],
                label="Example PCB Images",
                examples_per_page=10
            )

            generate_btn.click(
                fn=lambda img, prompt, upscale: self.generate_defect(img, None, prompt, upscale) if isinstance(img, dict) else self.tensor_to_pil(self.generate_defect(img, None, prompt, upscale)),
                inputs=[image_input, prompt_input, apply_upscaling],
                outputs=output_image
            )

        return interface

    def tensor_to_pil(self, tensor):
        """Convert output tensor to PIL Image"""
        tensor = tensor.squeeze().cpu().detach()
        return transforms.ToPILImage()(tensor)

def main():
    generator = PCBDefectGenerator()
    interface = generator.create_interface()
    interface.launch(share=True)

if __name__ == "__main__":
    main()