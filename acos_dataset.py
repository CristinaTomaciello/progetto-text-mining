import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer 

### Dataset per lo Step 1 (Multi-Task Extractor) che include anche le etichette per gli impliciti
class ACOSDataset(Dataset):
    '''Dataset personalizzato per il modello ACOS dello step 1 che include anche le etichette per gli impliciti.'''
    def __init__(self, df):
        self.input_ids = df['input_ids'].tolist()
        self.attention_mask = df['attention_mask'].tolist()
        self.labels = df['labels'].tolist()
        # Estraiamo le colonne per gli impliciti!
        self.implicit_aspect_label = df['implicit_aspect_label'].tolist()
        self.implicit_opinion_label = df['implicit_opinion_label'].tolist()

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.input_ids[idx], dtype=torch.long),
            'attention_mask': torch.tensor(self.attention_mask[idx], dtype=torch.long),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long),
            # Passiamo le etichette al Dataloader
            'implicit_aspect_labels': torch.tensor(self.implicit_aspect_label[idx], dtype=torch.long),
            'implicit_opinion_labels': torch.tensor(self.implicit_opinion_label[idx], dtype=torch.long)
        }

### Dataset per lo Step 2 (Cross-Encoder) che combina testo e contesto implicito
class ACOSPairDataset(Dataset):
    '''Dataset personalizzato per il modello ACOS dello step 2 che combina testo e contesto implicito.'''
    def __init__(self, df, tokenizer, max_length=128):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = row['review_text']
        
        # Estraiamo gli span (togliendo il +1 che avevamo messo per il [CLS] 
        # perché ora ci servono per tagliare la stringa originale)
        a_span = row['aspect_span']
        o_span = row['opinion_span']
        
        words = text.split()
        a_start, a_end = a_span[0] - 1, a_span[1] - 1
        o_start, o_end = o_span[0] - 1, o_span[1] - 1
        
        # Estraiamo le parole (o "null" se è implicito)
        aspect_str = " ".join(words[a_start:a_end]) if a_start >= 0 else "null"
        opinion_str = " ".join(words[o_start:o_end]) if o_start >= 0 else "null"
        
        # MAGIA CROSS-ENCODER: Creiamo la stringa contesto!
        cross_text = f"aspect: {aspect_str} opinion: {opinion_str}"

        # Il tokenizer unirà il 'text' e il 'cross_text' in automatico
        encoding = self.tokenizer(
            text,
            cross_text, # Passiamo la seconda stringa!
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        labels = torch.tensor(row['labels'], dtype=torch.long)

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': labels
            # Niente più aspect_span e opinion_span da passare!
        }