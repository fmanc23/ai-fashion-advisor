"""Main file.

@Author: Nicola Guerra
@Author: Davide Lupo
@Author: Francesco Mancinelli
"""
import json
from PIL import Image
from matplotlib import image, pyplot as plt
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader
import os

from src.similarity_calculator.cosine_similarity_function import CosineSimilarityFunction
from src.similarity_calculator.similarity_calculator import SimilarityFunction

from src.features_extractor.open_clip.model import OpenClipModel
from src.features_extractor.feature_extractor import FeatureExtractor

from src.ssd.nvidia_ssd_model import NVidiaSSDModel
from src.ssd.single_shot_detector import SingleShotDetector

from src.image_processor import ImageProcessor
from src.mask_processor import MaskProcessor

from src.segmentation.segformer_b2_clothes import SegformerB2Clothes
from src.segmentation.clothes_segmantion import ClothesSegmentation
from src.utils.utils import get_device

from src.datasets.polyvore import DatasetArguments, PolyvoreDataset
from src.models.load import load_model
import os
from src.model_args import Args
args = Args()

args.data_dir = f"{os.getcwd()}/dataset/polyvore_outfits"
args.checkpoint_dir = f"{os.getcwd()}/checkpoints"
args.model_path = f"{os.getcwd()}/checkpoints/outfit_transformer/cp/240608/AUC0.908.pth"

# Training Setting
args.num_workers = 4
args.test_batch_size = 128
args.with_cuda = True

file_path = f"{os.getcwd()}/dataset/polyvore_outfits"

def load_image(image_url: str) -> np.ndarray:
    return cv2.imread(image_url)

def ssd_detection(image_url: str) -> np.ndarray:
    ssd_model = SingleShotDetector(NVidiaSSDModel())
    return ssd_model.detect_person_in_image(image_url)

def segmentation(image : Image) -> torch.Tensor:
    segmentation_model = ClothesSegmentation(SegformerB2Clothes())
    return segmentation_model.apply_segmentation(image)

def delete_images(img_path: str, img_ext: str) -> None:
    files_to_delete = [
        img_path + "_resized" + img_ext,
        img_path + "_crop" + img_ext,
        img_path + "_segmented" + img_ext,
    ]

    for file_path in files_to_delete:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Deleted {file_path}")
        else:
            print(f"The file {file_path} does not exist")

def mapping_label_to_mask(masks: dict) -> list:
    feature_extractor = FeatureExtractor(OpenClipModel())
    with open("./prompts_no_desc.json", "r") as prompts_file:
        prompts = json.load(prompts_file).get('prompts')

    map_label_mask_list = []
    for i, mask in enumerate(masks.values()):
        img_tensor = feature_extractor.model.load_and_process_image(mask)
        inferences = feature_extractor.model.run_inference(img_tensor, prompts)
        _, index_label = inferences.squeeze(0).max(dim=0)

        with open("./mapping.json", "r") as mappings_file:
            mappings = json.load(mappings_file)

        for label, idx_list in mappings.items():
            if index_label.item() in idx_list:
                map_label_mask_list.append({
                    "idx_label": i,
                    "label": label,
                    "mask": mask
                })
            
    return map_label_mask_list

def features_extraction(masks: dict) -> list:
    feature_extractor = FeatureExtractor(OpenClipModel())
    return feature_extractor.extract_masks_features(masks)

def calculate_similarity(features) -> torch.Tensor:
    similarity_function = SimilarityFunction(CosineSimilarityFunction())
    return similarity_function.compute_similarity(features)

def find_index_mask_to_replace(masks: dict) -> int:
    features = features_extraction(masks)
    similarity_matrix = calculate_similarity(features)
    mean_similarity = similarity_matrix.mean(dim=0)
    _, index_lowest_similarity = mean_similarity.squeeze(0).min(dim=0)
    return index_lowest_similarity

def cp_evaluation(model,dataloader,device): 
    for iter, batch in enumerate(dataloader, start=1):
        inputs = {key: value.to(device) for key, value in batch['inputs'].items()}
        input_embeddings = model.batch_encode(inputs)            
        score = model.get_score(input_embeddings)
        return score

def fitb_evaluation(model, dataloader, device):
    total_scores = []
    for _, batch in enumerate(dataloader, start=1):
        questions = {key: value.to(device) for key, value in batch['questions'].items()}
        candidates = {key: value.to(device) for key, value in batch['candidates'].items()}
        #print(batch['candidates'].items())

        question_item_embeds = model.batch_encode(questions)           
        question_embeds = model.get_embedding(question_item_embeds) # B, EMBEDDING_DIM
        
        candidate_item_embeds = model.batch_encode(candidates) # B, N_CANDIDATES(1 positive, 3 negative), EMBEDDING_DIM
        B, N_CANDIDATES = candidates['mask'].shape

        candidate_item_embeds['mask'] = candidate_item_embeds['mask'].view(B * N_CANDIDATES, -1)
        candidate_item_embeds['embeds'] = candidate_item_embeds['embeds'].view(B * N_CANDIDATES, 1, -1)
        candidate_embeds = model.get_embedding(candidate_item_embeds).view(B, N_CANDIDATES, -1) # B, N_CANDIDATES, EMBEDDING_DIM

        with torch.no_grad():
            scores = torch.sum(question_embeds.unsqueeze(1).detach() * candidate_embeds.detach(), dim=-1)
            total_scores.append(scores)

    return total_scores

def main():
    """
    Funzione principale dello script. Esegue i seguenti passaggi:

    Nota:   Questa funzione non restituisce nulla.
            Salva i risultati intermedi e finali su disco e mostra il risultato
    """
    img_path = "./static/image_test_2"
    img_ext = ".jpg"

    #salvataggio dimensione immagine di input
    input_image =  load_image(img_path + img_ext)

    # Denoise dell'immagine
    denoise_image = ImageProcessor.denoise_image(input_image)

    # Ridimensiona l'immagine
    resized_image = ImageProcessor.resize_image(denoise_image, (300, 300))
    cv2.imwrite(img_path + "_resized" + img_ext, resized_image)

    # Esegue la rilevazione della persona
    detected_image = ssd_detection(img_path + "_resized" + img_ext)
    detected_image_pil = Image.fromarray(detected_image.astype(np.uint8))
    detected_image_pil.save(img_path + "_crop" + img_ext)

    # Applica la segmentazione dei vestiti
    segmented_image = segmentation(detected_image_pil)
    Image.fromarray(segmented_image.numpy().astype(np.uint8)).save(img_path+"_segmented"+img_ext)
    segmented_image = segmented_image.numpy().astype(np.uint8)

    # Rimozione immagini temporanee
    delete_images(img_path, img_ext)

    # Applica le maschere
    masks = MaskProcessor.compute_masks(detected_image, segmented_image)

    # inferenza maschere con classi open clip
    map_label_mask_list = mapping_label_to_mask(masks)

    # selezione maschera da rimpiazzare
    idx_mask_to_replace = find_index_mask_to_replace(masks)
    mask_to_replace = map_label_mask_list[idx_mask_to_replace.item()]

    # salvataggio su disco delle maschere
    mask_values = masks.values()
    for i, mask in enumerate(mask_values):
        cv2.imwrite(f"{os.getcwd()}/dataset/polyvore_outfits/images/{i}.jpg", mask)

    # Define the new element you want to add
    items = []
    for i in range(len(mask_values)):
        items.append({ "item_id": str(i), "index": i+1 })

    new_element = [{ "items": items, "set_id": "222" }]

    file = file_path + '/nondisjoint/outfit_to_score.json'
    with open(file, 'w', encoding='utf-8') as f:
        json.dump(new_element, f, ensure_ascii=False, indent=4)

    # Define the new row you want to add
    new_row = '1 '
    for i in range(len(mask_values)):
        new_row += '222_' + str(i+1) + ' '

    # Write the manipulated JSON string to the file
    file = file_path + '/nondisjoint/compatibility_outfit_to_score.txt'
    with open(file, 'w') as f:
        f.write(new_row) 

    # Load the JSON data from the file
    file = file_path + '/polyvore_item_metadata.json'
    with open(file, 'r') as f:
        data = json.load(f)

    category_to_retrieve = mask_to_replace["label"]

    for i, mask in enumerate(map_label_mask_list):
        new_element_value = {
            "url_name": "", 
            "description": "", 
            "categories": "", 
            "title": "", 
            "related": "", 
            "category_id": mask["idx_label"], 
            "semantic_category": mask["label"]
        }
        data[str(i)] = new_element_value

    # Save the updated JSON data back to the file
    with open(file, 'w') as f:
        json.dump(data, f, indent=4)

    #Outfit-Transformer
    cuda_condition = torch.cuda.is_available() and args.with_cuda
    device = torch.device("cuda:0" if cuda_condition else "cpu")

    args.model_path = f"{os.getcwd()}/checkpoints/outfit_transformer/cir/240719/ACC0.962.pth"

    model, input_processor = load_model(args)
    model.to(device)

    items = []
    for i in range(len(mask_values)):
        items.append({ 'item_id' : str(i), 'index' : i+1 })
        
    file = file_path + '/nondisjoint/all_outfit.json'
    with open(file, 'r') as f:  
        data = json.load(f)
          
    data[0] = {'items' : items, 'set_id' : 222} 

    with open(file, 'w') as f:
        json.dump(data, f, indent=4)

    file = file_path + '/nondisjoint/categories.json'
    with open(file, 'r') as f:
        data = json.load(f)

    question = []
    for i in range(len(mask_values)):
        if i != idx_mask_to_replace:
            question.append('222_'+str(i+1))
    
    # Calcolo del numero di gruppi da creare basato sulla lunghezza della lista e sulla dimensione del gruppo (12)
    polyvore_answer_list = data[category_to_retrieve]
    num_groups = 200 #len(polyvore_answer_list) // 12
    data_to_write = [
        {
            'question': question,
            'answers': polyvore_answer_list[i*12:12+i*12]
        }
        for i in range(num_groups)
    ]

    file = file_path + '/nondisjoint/fill_in_blank_all_outfit.json'
    with open(file, 'w') as f:
        json.dump(data_to_write, f, indent=4)

    test_dataset_args = DatasetArguments(
        polyvore_split = args.polyvore_split,
        task_type = 'fitb',
        dataset_type = 'all_outfit'
    )

    test_dataset = PolyvoreDataset(
        data_dir = args.data_dir,
        args = test_dataset_args,
        input_processor = input_processor
    )

    test_dataloader = DataLoader(
        dataset = test_dataset,
        batch_size = args.test_batch_size,
        shuffle = False,
        num_workers = args.num_workers
    )

    model.eval()
    with torch.no_grad():
        scores = fitb_evaluation(
            model = model,
            dataloader = test_dataloader,
            device = device
        )

    concat_tensor = torch.clone(scores[0])
    for i in range(len(scores)-1):
        concat_tensor = torch.cat((concat_tensor, scores[i+1]), dim=0)

    flattened_tensor = concat_tensor.flatten()
    index = flattened_tensor.argmax()

    file = file_path + '/nondisjoint/categories.json'
    with open(file, 'r') as f:
        data = json.load(f)

    index_max = flattened_tensor.argmax()
    set_id_index = data[category_to_retrieve][index_max]
    set_id = set_id_index.split('_')[0]
    index = set_id_index.split('_')[1]

    file = file_path + '/nondisjoint/all_outfit.json'
    with open(file, 'r') as f:
        data = json.load(f)

    for i in range(len(data)):
        if data[i]['set_id'] == set_id:
            items = data[i]['items']
            for j in range(len(items)):
                if items[j]['index'] == int(index):
                    img_ret = items[j]['item_id']
            break
    
    print(img_ret)
    plt.imshow(cv2.imread(f"{os.getcwd()}/dataset/polyvore_outfits/images/{img_ret}.jpg"))
    plt.show()

if __name__ == "__main__":
    main()
