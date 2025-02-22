# dataset_loader.py
import datasets
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import json
import os
from datetime import datetime
import pandas as pd

class PCBDataset(Dataset):
    def __init__(self, split="train"):
        # Load the dataset from Hugging Face
        self.dataset = datasets.load_dataset("Thanarit/PCB")[split]
        
        # Convert to image format when loading
        self.dataset = self.dataset.cast_column("master_image", datasets.Image())
        self.dataset = self.dataset.cast_column("defect_image", datasets.Image())
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        return {
            'pair_id': item['pair_id'],
            'master_image': item['master_image'],
            'defect_image': item['defect_image']
        }
    

import gradio as gr
import numpy as np
from PIL import Image
import io
from datasets import Dataset

class LabelingSession:
    def __init__(self):
        self.save_dir = "labeling_progress"
        os.makedirs(self.save_dir, exist_ok=True)
        self.session_file = os.path.join(self.save_dir, "session.json")
        self.masks_dir = os.path.join(self.save_dir, "masks")
        os.makedirs(self.masks_dir, exist_ok=True)
        
        if os.path.exists(self.session_file):
            with open(self.session_file, 'r') as f:
                self.labeled_data = json.load(f)
        else:
            self.labeled_data = []
        
        self.current_index = len(self.labeled_data)
    
    def save_session(self):
        with open(self.session_file, 'w') as f:
            json.dump(self.labeled_data, f)
    
    def add_label(self, item):
        mask_filename = f"mask_{len(self.labeled_data)+1}.png"
        mask_path = os.path.join(self.masks_dir, mask_filename)
        item['mask'].save(mask_path)
        
        metadata = {
            'pair_id': item['pair_id'],
            'master_image_path': self._save_image(item['master_image'], f"master_{len(self.labeled_data)+1}.png"),
            'defect_image_path': self._save_image(item['defect_image'], f"defect_{len(self.labeled_data)+1}.png"),
            'mask_path': mask_path,
            'description': item['description']
        }
        
        self.labeled_data.append(metadata)
        self.save_session()
    
    def _save_image(self, image, filename):
        path = os.path.join(self.save_dir, filename)
        # Convert numpy array to PIL Image if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        image.save(path)
        return path
    
    def get_preview_data(self):
        return pd.DataFrame(self.labeled_data)

def create_mask_interface():
    dataset = PCBDataset("train")
    session = LabelingSession()  # Replace labeled_data with session
    
    def load_next_pair():
        nonlocal dataset
        if session.current_index < len(dataset):
            item = dataset[session.current_index]
            session.current_index += 1
            return (
                item['master_image'],
                item['defect_image'],
                "",
                update_preview()
            )
        return None, None, "", update_preview()

    def save_label(master_img, defect_img_with_mask, description):
        if defect_img_with_mask is not None:
            # Convert to PIL Image consistently
            mask_array = np.array(defect_img_with_mask["mask"])
            binary_mask = (mask_array > 128).astype(np.uint8) * 255
            mask_pil = Image.fromarray(binary_mask)
            mask_pil = mask_pil.resize((650, 650))

            # Convert defect image to PIL if needed
            defect_img = defect_img_with_mask["image"]
            if isinstance(defect_img, np.ndarray):
                defect_img = Image.fromarray(defect_img)

            # Check for edit mode
            if hasattr(session, 'current_edit_index'):
                row_idx = session.current_edit_index
                session.labeled_data[row_idx].update({
                    'mask': mask_pil,
                    'description': description
                })
                session.save_session()
                del session.current_edit_index
                return "Label updated successfully!", gr.Image.update(value=defect_img_with_mask["image"]), update_preview()
            else:
                new_entry = {
                    'pair_id': f"{len(session.labeled_data) + 1}",
                    'master_image': master_img,
                    'defect_image': defect_img,
                    'mask': mask_pil,
                    'description': description
                }
                session.add_label(new_entry)
                return "Label saved successfully!", gr.Image.update(value=defect_img_with_mask["image"]), update_preview()
        return "No mask drawn!", None, update_preview()

    def update_preview():
        df = session.get_preview_data()
        return gr.DataFrame.update(value=df)

    def edit_row(evt: gr.SelectData):
        row_idx = evt.index[0]
        row_data = session.labeled_data[row_idx]
        master_img = Image.open(row_data['master_image_path'])
        defect_img = Image.open(row_data['defect_image_path'])
        mask = Image.open(row_data['mask_path'])
        
        # Store edit index in session
        session.current_edit_index = row_idx
        
        # Return as sketch tool compatible format
        return {
            "image": np.array(defect_img),  # Convert to numpy array
            "mask": np.array(mask)
        }, row_data['description']

    with gr.Blocks(css="""
        .container {max-width: 1100px !important}
        #defect_image {min-height: 400px}
        #defect_image [data-testid="image"], #defect_image [data-testid="image"] > div{min-height: 400px}
    """) as interface:
        gr.Markdown("""
        # PCB Defect Labeling Tool
        ### Instructions:
        1. View the master (normal) PCB on the left
        2. On the defective PCB image on the right:
           - Draw directly on defective areas using the sketch tool
           - Use the eraser to correct mistakes
        3. Describe the defects you've marked
        4. Click Save Label when done
        """)
        
        with gr.Row():
            master_image = gr.Image(
                label="Master PCB", 
                interactive=False,
                height=650,
                width=650
            )
            defect_image = gr.Image(
                label="Draw on Defective Areas",
                source="upload",
                tool="sketch",
                type="pil",
                elem_id="defect_image",
                height=650,
                width=650
            )
        
        with gr.Row():
            description = gr.Textbox(
                label="Defect Description",
                placeholder="Describe the defects you've marked...",
                lines=3
            )
            
        with gr.Row():
            next_btn = gr.Button("Next Image Pair")
            save_btn = gr.Button("Save Label", variant="primary")
            push_btn = gr.Button("Push to HuggingFace")
            
        status = gr.Textbox(label="Status")

        # Add preview section
        gr.Markdown("### Labeled Dataset Preview")
        preview = gr.DataFrame(
            session.get_preview_data(),
            interactive=False,
            label="Labeled Images"
        )

        # Event handlers
        next_btn.click(
            load_next_pair,
            outputs=[master_image, defect_image, description]
        )
        
        save_btn.click(
            save_label,
            inputs=[master_image, defect_image, description],
            outputs=[status, defect_image, preview]
        )

        # Move this block UP before the push_btn.click handler
        def push_to_huggingface():
            if not session.labeled_data:
                return "No data to push! Please label some images first."
            
            dataset_dict = {
                'pair_id': [item['pair_id'] for item in session.labeled_data],
                'master_image': [Image.open(item['master_image_path']) for item in session.labeled_data],
                'defect_image': [Image.open(item['defect_image_path']) for item in session.labeled_data],
                'mask': [Image.open(item['mask_path']) for item in session.labeled_data],
                'description': [item['description'] for item in session.labeled_data]
            }
            
            dataset = Dataset.from_dict(dataset_dict)
            dataset = dataset.cast_column("master_image", datasets.Image())
            dataset = dataset.cast_column("defect_image", datasets.Image())
            dataset = dataset.cast_column("mask", datasets.Image())
            
            dataset.push_to_hub("Thanarit/PCB-v3")
            return "Dataset pushed to Hugging Face successfully!"

        # Now this will reference the already-defined function
        push_btn.click(
            push_to_huggingface,
            outputs=[status]
        )

        # Update event handlers
        preview.select(
            edit_row, 
            outputs=[defect_image, description]
        )

    return interface

if __name__ == "__main__":
    interface = create_mask_interface()
    interface.launch(share=True)