import gradio as gr
import torch
from PIL import Image
import numpy as np
from datasets import load_dataset
import os
from inpaintinference import PCBLatentInpaintingInference
import torchvision.transforms as transforms
from torchvision.utils import save_image

class PCBDefectGenerator:
    def __init__(self, model_path="working_model/checkpoint-1999-126images"):
        self.model = PCBLatentInpaintingInference(model_path)
        self.dataset = load_dataset("Thanarit/PCB-v3")["train"]
        
        # Create examples directory if it doesn't exist
        self.examples_dir = "example_images"
        os.makedirs(self.examples_dir, exist_ok=True)
        
        # Prepare example images
        self.examples = self.prepare_examples()

    def prepare_examples(self):
        examples = []
        for idx in range(min(10, len(self.dataset))):
            example = self.dataset[idx]
            if example["description"] != "nothing":
                # Save image to file
                img_path = os.path.join(self.examples_dir, f"example_{idx}.png")
                example["master_image"].save(img_path)
                examples.append([
                    img_path,  # Path to saved image
                    example["description"]  # Description
                ])
        return examples

    # def generate_defect_old(self, master_image, mask_image, prompt):
    #     """
    #     Generate defective PCB image based on mask and prompt
    #     """
    #     # Handle input from Gradio sketch tool
    #     if isinstance(master_image, dict):
    #         base_image = master_image["image"]
    #         mask = master_image["mask"]
    #     else:
    #         base_image = master_image
    #         mask = mask_image["mask"] if mask_image is not None else None

    #     # Convert numpy array to PIL Image if needed
    #     if isinstance(base_image, np.ndarray):
    #         base_image = Image.fromarray(base_image)
        
    #     # Convert to RGB and transform exactly like inpaintinference.py
    #     base_image = base_image.convert("RGB")
    #     transform_image = transforms.Compose([
    #         transforms.Resize((512, 512)),
    #         transforms.ToTensor(),
    #         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    #     ])
    #     master_tensor = transform_image(base_image).unsqueeze(0)

    #     # Process mask
    #     if mask is None:
    #         mask = torch.zeros((1, 1, 512, 512), device=master_tensor.device)
    #     else:
    #         if isinstance(mask, np.ndarray):
    #             mask = Image.fromarray(mask)
    #         mask = mask.convert("L")  # Convert to grayscale
    #         mask = transforms.Compose([
    #             transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.NEAREST),
    #             transforms.ToTensor()
    #         ])(mask)
    #         mask = (mask > 0.5).float()  # Binary mask
    #         if mask.ndim == 3 and mask.shape[0] == 3:
    #             mask = mask[0].unsqueeze(0)
    #         mask = mask.unsqueeze(0)

    #     # Generate defective image
    #     with torch.cuda.amp.autocast(enabled=True):
    #         generated = self.model.generate(
    #             prompt=prompt,
    #             master_image=master_tensor,
    #             mask=mask,
    #             num_inference_steps=50
    #         )[0]

    #     # Post-process output
    #     generated = (generated / 2 + 0.5).clamp(0, 1)
    #     generated = (generated.cpu().permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    #     generated_image = Image.fromarray(generated)

    #     return generated_image
    
    def generate_defect(self, master_image, mask_image, prompt):
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
            num_inference_steps=50
        )

        save_image(generated_image, "generated_image.png")

        return generated_image
    

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
                examples_per_page=5
            )

            generate_btn.click(
                fn=lambda *args: self.tensor_to_pil(self.generate_defect(*args)),
                inputs=[image_input, gr.State(None), prompt_input],
                outputs=output_image
            )

        return interface

    def tensor_to_pil(self, tensor):
        """Convert output tensor to PIL Image"""
        tensor = tensor.squeeze().cpu().detach()
        # tensor = (tensor * 0.5) + 0.5  # Undo normalization
        # tensor = tensor.clamp(0, 1)
        return transforms.ToPILImage()(tensor)

def main():
    generator = PCBDefectGenerator()
    interface = generator.create_interface()
    interface.launch(share=True)

if __name__ == "__main__":
    main()