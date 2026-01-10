import logging
from pathlib import Path
logger = logging.getLogger(__name__)


class ProjectPaths:
    """Centralized path management for the flood detection project"""

    def __init__(self, project_path: Path):
        self.project_path = project_path
        self._image_code = None
        
        # Main working directories
        self.dataset_path = self.project_path / "data" /  "4final" / "dataset"
        self.training_path = self.project_path / "data" / "4final" / "training"
        self.predictions_path = self.project_path / "data" / "4final" / 'predictions'
        self.test_path = self.project_path / "data" / "4final" / "testing"
        
        # Data subdirectories
        self.images_path = self.dataset_path / 'S1Hand'
        self.labels_path = self.dataset_path / 'LabelHand'
        
        # CSV files
        self.train_csv = self.dataset_path / "flood_train_data.csv"
        self.val_csv = self.dataset_path / "flood_valid_data.csv"
        self.test_csv = self.dataset_path / "flood_test_data.csv"
        
        # Checkpoint directories (consolidated)
        self.ckpt_input_path = project_path / "checkpoints" / 'ckpt_input'
        self.ckpt_training_path = self.project_path / "checkpoints" / "ckpt_training"
        
        # Config files
        self.main_config = project_path / "configs" / "floodaiv2_config.yaml"
        self.minmax_config = project_path / "configs" / "global_minmax_INPUT" / "global_minmax.json"
        
        # Environment file
        self.env_file = self.project_path / ".env"

    @property
    def image_code(self) -> str:
        if self._image_code is None:  
            predict_input = self.project_path / "data" / "4final" / "predict_input"
            # DEFINE PREDICT_INPUT
            if not predict_input.exists():
                raise FileNotFoundError(f"Predict input not found: {predict_input}")
            # FIND THE INPUT IMAGE TO EXTRACT IMAGE CODE
            input_names = [f for f in predict_input.iterdir() 
                           if f.is_file() 
                           and not f.name.startswith('.') 
                           and f.suffix.lower() in ['.tif', '.tiff']]
            if input_names:
                # extract file name
                splits = input_names[0].stem.split('_')
                raw_code = '_'.join(splits[:2])
                # Sanitize: replace colons with hyphens or underscores 
                self._image_code = raw_code.replace(':', '-')  # ← ADD THIS   
     

            # OTHERWISE GET IT FROM THE TILE FOLDER NAME
            else:
                logger.info(f"No '.tif' imput image found in {predict_input}\nso looking  in tile folder name in {self.predictions_path} for image_code...")
   

                tile_folders = [f for f in self.predictions_path.iterdir() if not f.name.startswith('.') and  f.is_dir() and "tiles" in f.name.lower()]

                if not tile_folders:
                    raise FileNotFoundError(f"No input files or tile folders found in {self.predictions_path}")
                # Extract image code from folder name
                # e.g., "Ghana_313799_tiles" -> "Ghana_313799"
                folder_name = tile_folders[0].name
                self._image_code = folder_name.replace("_tiles", "")
                logger.info(f"Extracted image_code from pre-tiled folder: {self._image_code}")
        return self._image_code

 

    
    def get_inference_paths(self, tile_size: int =512, threshold: float = 0.5, output_filename: str = '_name') -> dict:
        # GRAB OUTPUT_FILENAME
      
        image_tiles_path = self.predictions_path / f'tiles'
        return {
            'predict_input_path': self.project_path / "data" / "4final" / "predict_input",
            'pred_tiles_path': self.predictions_path / f'{output_filename}_predictions',
            'image_tiles_path': image_tiles_path,
            'extracted_path': self.predictions_path / 'extracted',
            'file_list_csv_path': self.predictions_path / "predict_tile_list.csv",
            'stitched_image': self.predictions_path / f'{self.image_code}_{tile_size}_{threshold}_{output_filename}_WATER_AI.tif',
            'metadata_path': image_tiles_path / 'tile_metadata.json'
        }
    
    def get_training_paths(self):
        """Get training/testing specific paths"""
        return {
            'image_tiles_path': self.dataset_path,
            'metadata_path': self.dataset_path / 'tile_metadata_pth.json'
        }
    
    def validate_paths(self, job_type: str, ckpt_input: bool = False):
        """Validate that required paths exist for the given job type"""
        errors = []
        required_paths = []
        if job_type in ('train', 'test'):
            required_paths = [
                (self.dataset_path, "Dataset directory"),
                (self.images_path, "Images directory"),
                (self.labels_path, "Labels directory"),
            ]
            
            if job_type == 'train':
                required_paths.extend([
                    (self.train_csv, "Training CSV"),
                    (self.val_csv, "Validation CSV")
                ])
            elif job_type == 'test':
                required_paths.append((self.test_csv, "Test CSV"))
                
        elif job_type == 'inference':
            # Inference paths are created dynamically, less validation needed
            pass
            
        # Always check checkpoint folder, but pick which one based on flag/job
        if job_type ==  ckpt_input:
            ckpt_dir = self.ckpt_input_path
            ckpt_desc = "Checkpoint input directory"
        else:
            ckpt_dir = self.ckpt_training_path
            ckpt_desc = "Checkpoint training directory"

        required_paths.append((ckpt_dir, ckpt_desc))
        
        for path, description in required_paths:
            if not path.exists():
                errors.append(f"{description} not found: {path}")
        
        # Check for checkpoint files
        if not any(ckpt_dir.rglob("*.ckpt")):
            errors.append(f"No checkpoint files found in: {ckpt_dir}")
            
        return errors