"""
@author https://github.com/owj0421/OutfitTransformer
"""
import os
from datetime import datetime
from typing import Tuple
import torch
import wandb
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim import AdamW
from tqdm import tqdm

from src.utils.utils import get_device, save_model

from src.models.recommender import RecommendationModel
from src.models.load import load_model

from src.datasets.polyvore import DatasetArguments, PolyvoreDataset

from src.loss.info_nce import InfoNCE

from src.model_args import Args
args = Args()

args.data_dir = f"{os.getcwd()}/dataset/polyvore_outfits"
args.checkpoint_dir = f"{os.getcwd()}/checkpoints"
args.model_path = f"{os.getcwd()}/checkpoints/outfit_transformer/cir/240610/ACC0.647.pth"

# Training Setting
args.n_epochs = 30
args.num_workers = 4
args.train_batch_size = 128
args.val_batch_size = 128
args.lr = 4e-5
args.wandb_key = None
args.use_wandb = bool(args.wandb_key)
args.with_cuda = True

def cir_iteration(
    epoch: int,
    model_instance: RecommendationModel,
    optimizer: AdamW,
    scheduler: OneCycleLR,
    dataloader: DataLoader,
    device: torch.device,
    is_train: bool,
    use_wandb: bool
) -> Tuple[float, float]:
    """
    Esegue un'iterazione di addestramento o validazione sull'intero dataset.

    Questa funzione elabora il dataset in batch utilizzando il dataloader fornito,
    aggiorna i parametri del modello durante l'addestramento.
    Opzionalmente registra le metriche su Weights & Biases.
    
    Args:
    ----------
        epoch (int): numero epoche
        model_instance (RecommendationModel): istanza modello
        optimizer (AdamW): ottimizzatore
        scheduler (OneCycleLR): scheduler
        dataloader (DataLoader): classe per il dataloader
        device (torch.device): device per addestramento, cuda o cpu
        is_train (bool): flag per checkpoint
        use_wandb (bool): flag per wandb

    Returns:
    ----------
        Tuple[float, float]: loss e accuracy
    """
    criterion = InfoNCE(negative_mode="unpaired")

    type_str = "cir train" if is_train else "cir valid"
    epoch_iterator = tqdm(dataloader)

    loss = 0.
    total_y_true = []
    total_y_pred = []

    for iter, batch in enumerate(epoch_iterator, start=1):
        anchor = {key: value.to(device) for key, value in batch['anchor'].items()}
        positive = {key: value.to(device) for key, value in batch['positive'].items()}

        anchor_item_embeds = model_instance.batch_encode(anchor)
        anchor_embeds = model_instance.get_embedding(anchor_item_embeds) # B, EMBEDDING_DIM

        positive_item_embeds = model_instance.batch_encode(positive)
        positive_embeds = model_instance.get_embedding(positive_item_embeds) # B, EMBEDDING_DIM

        running_loss = criterion(
            query=anchor_embeds,
            positive_key=positive_embeds,
        )

        loss += running_loss.item()
        if is_train:
            optimizer.zero_grad()
            running_loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()

        with torch.no_grad():
            logits = anchor_embeds @ positive_embeds.transpose(-2, -1)
            labels = torch.arange(len(anchor_embeds), device=anchor_embeds.device)

            total_y_true.append(labels.cpu())
            total_y_pred.append(torch.argmax(logits, dim=-1).cpu())

        is_correct = total_y_true[-1] == total_y_pred[-1]
        running_acc = torch.sum(is_correct).item() / torch.numel(is_correct)
        epoch_iterator.set_description(
            f'Loss: {running_loss:.5f} | Acc: {running_acc:.3f}'
        )

        if use_wandb:
            log = {
                f'{type_str}_loss': running_loss, 
                f'{type_str}_acc': running_acc, 
                f'{type_str}_step': epoch * len(epoch_iterator) + iter
            }
            if is_train and scheduler is not None:
                log["learning_rate"] = scheduler.get_last_lr()[0]
            wandb.log(log)

    loss = loss / iter
    total_y_true = torch.cat(total_y_true)
    total_y_pred = torch.cat(total_y_pred)
    is_correct = total_y_true == total_y_pred
    acc = torch.sum(is_correct).item() / torch.numel(is_correct)
    print(
        f'[{type_str} END] Epoch: {epoch + 1:03} | loss: {loss:.5f} | Acc: {acc:.3f}'
    )

    return loss, acc

def fitb_iteration(
    epoch: int,
    model: RecommendationModel,
    optimizer: AdamW,
    scheduler: OneCycleLR,
    dataloader: DataLoader,
    device: torch.device,
    is_train: bool,
    use_wandb: bool
) -> Tuple[float, float]:
    """
    Esegue un'iterazione di addestramento o validazione per il task FITB (Fill In The Blank).

    Questa funzione elabora il dataset in batch utilizzando il dataloader fornito,
    aggiorna i parametri del modello durante l'addestramento.
    Opzionalmente registra le metriche su Weights & Biases.

    Args:
    ----------
        epoch (int): numero epoche
        model_instance (RecommendationModel): istanza modello
        optimizer (AdamW): ottimizzatore
        scheduler (OneCycleLR): scheduler
        dataloader (DataLoader): classe per il dataloader
        device (torch.device): device per addestramento, cuda o cpu
        is_train (bool): flag per checkpoint
        use_wandb (bool): flag per wandb

    Returns:
        Tuple[float, float]: loss e accuracy
    """
    criterion = InfoNCE(negative_mode='paired')

    type_str = "fitb train" if is_train else "fitb valid"
    epoch_iterator = tqdm(dataloader)

    loss = 0.
    total_y_true = []
    total_y_pred = []

    for iter, batch in enumerate(epoch_iterator, start=1):
        questions = {key: value.to(device) for key, value in batch['questions'].items()}
        candidates = {key: value.to(device) for key, value in batch['candidates'].items()}

        question_item_embeds = model.batch_encode(questions)
        question_embeds = model.get_embedding(question_item_embeds) # B, EMBEDDING_DIM

        candidate_item_embeds = model.batch_encode(candidates) # B, N_CANDIDATES(1 pos, 3 neg)
        b, n_candidates = candidates['mask'].shape
        candidates = b * n_candidates
        candidate_item_embeds['mask'] = candidate_item_embeds['mask'].view(candidates, -1)
        candidate_item_embeds['embeds'] = candidate_item_embeds['embeds'].view(candidates, 1, -1)

        candidate_embeds = model.get_embedding(candidate_item_embeds).view(b, n_candidates, -1)
        pos_candidate_embeds = candidate_embeds[:, 0, :]
        neg_candidate_embeds = candidate_embeds[:, 1:, :]

        running_loss = criterion(
            query=question_embeds,
            positive_key=pos_candidate_embeds,
            negative_keys=neg_candidate_embeds
        )

        loss += running_loss.item()
        if is_train:
            optimizer.zero_grad()
            running_loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()

        with torch.no_grad():
            question_embeds_detach = question_embeds.unsqueeze(1).detach()
            candidates_embeds_detach = candidate_embeds.detach()
            scores = torch.sum(question_embeds_detach * candidates_embeds_detach, dim=-1)
            total_y_true.append(torch.zeros(b, dtype=torch.long).cpu())
            total_y_pred.append(torch.argmax(scores, dim=-1).cpu())

        is_correct = total_y_true[-1] == total_y_pred[-1]
        running_acc = torch.sum(is_correct).item() / torch.numel(is_correct)
        epoch_iterator.set_description(
            f"Loss: {running_loss:.5f} | Acc: {running_acc:.3f}"
        )

        if use_wandb:
            log = {
                f"{type_str}_loss": running_loss, 
                f"{type_str}_acc": running_acc, 
                f"{type_str}_step": epoch * len(epoch_iterator) + iter
            }
            if is_train and scheduler is not None:
                log["learning_rate"] = scheduler.get_last_lr()[0]

            wandb.log(log)

    loss = loss / iter
    total_y_true = torch.cat(total_y_true)
    total_y_pred = torch.cat(total_y_pred)
    is_correct = total_y_true == total_y_pred
    acc = torch.sum(is_correct).item() / torch.numel(is_correct)
    print(f"[{type_str} END] Epoch: {epoch + 1:03} | loss: {loss:.5f} | Acc: {acc:.3f}")

    return loss, acc


if __name__ == "__main__":
    TASK = "cir"
    EMBEDDER_TYPE = "outfit_transformer" if not args.use_clip_embedding else "clip"

    # Wandb
    if args.use_wandb:
        os.environ["WANDB_API_KEY"] = args.wandb_key
        os.environ["WANDB_PROJECT"] = f"OutfitTransformer-{TASK}"
        os.environ["WANDB_LOG_MODEL"] = "all"
        wandb.login()
        run = wandb.init()

    date_info = datetime.today().strftime("%y%m%d")
    save_dir = os.path.join(args.checkpoint_dir, EMBEDDER_TYPE, TASK, date_info)

    cuda_condition = torch.cuda.is_available() and args.with_cuda
    device = torch.device(get_device())

    model, input_processor = load_model(args)
    model.to(device)

    train_dataset_args = DatasetArguments(
        polyvore_split=args.polyvore_split,
        task_type="cir",
        dataset_type="train"
    )
    train_dataset = PolyvoreDataset(
        args.data_dir,
        train_dataset_args,
        input_processor
    )
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )

    #val_dataset_args = DatasetArguments(
    #    polyvore_split=args.polyvore_split,
    #    task_type="fitb",
    #    dataset_type="valid"
    #)

    #val_dataset = PolyvoreDataset(args.data_dir, val_dataset_args, input_processor)
    #val_dataloader = DataLoader(
    #    dataset=val_dataset,
    #    batch_size=args.val_batch_size,
    #    shuffle=False,
    #    num_workers=args.num_workers
    #)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = OneCycleLR(
        optimizer,
        args.lr,
        epochs=args.n_epochs,
        steps_per_epoch=len(train_dataloader)
    )

    best_acc = 0
    for epoch in range(args.n_epochs):
        model.train()
        train_loss, train_acc = cir_iteration(
            epoch, model, optimizer, scheduler,
            dataloader=train_dataloader, device=device, is_train=True, use_wandb=args.use_wandb
        )
        #model.eval()
        #with torch.no_grad():
        #    val_loss, val_acc = fitb_iteration(
        #        epoch, model, optimizer, scheduler,
        #        dataloader=val_dataloader, device=device, is_train=False, use_wandb=args.use_wandb
        #    )
        
        model_name = f'ACC{train_acc:.3f}'
        save_model(model, save_dir, model_name, device)
        #if val_acc >= best_acc:
        #    best_acc = val_acc
        #    model_name = f'ACC{val_acc:.3f}'
        #    save_model(model, save_dir, model_name, device)
