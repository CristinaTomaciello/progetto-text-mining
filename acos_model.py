import torch
import torch.nn as nn
from transformers import AutoModel
from torchcrf import CRF

#FASE 1: Estrazione di Aspetti e Opinioni (ACOS)
class ModernBertACOS_Extractor(nn.Module):
    def __init__(self, model_name="answerdotai/ModernBERT-base", num_labels=5):
        super().__init__()
        # Carichiamo la "schiena" del modello (l'encoder base)
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        
        # Testolina 1: Emette i "punteggi grezzi" per il CRF
        self.token_classifier = nn.Linear(hidden_size, num_labels)
        
        # IL LAYER CRF (Correttore Ortografico per Sequenze)
        self.crf = CRF(num_labels, batch_first=True)
        
        # Testoline 2 e 3: Indovinano se ci sono impliciti (Manteniamo la tua intuizione!)
        self.implicit_aspect_classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 2)
        )
        self.implicit_opinion_classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 2)
        )
        
    def forward(self, input_ids, attention_mask, labels=None, implicit_aspect_labels=None, implicit_opinion_labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state 
        
        # Prendiamo il token [CLS] (posizione 0) per i classificatori binari
        cls_output = sequence_output[:, 0, :] 
        
        # Punteggi grezzi (emissions) per il CRF
        emissions = self.token_classifier(sequence_output)
        
        # Le due testoline calcolano i logit per gli impliciti
        imp_asp_logits = self.implicit_aspect_classifier(cls_output)
        imp_opi_logits = self.implicit_opinion_classifier(cls_output)
        
        loss = None
        
        # --- FASE DI ADDESTRAMENTO (Calcolo della Loss) ---
        if labels is not None and implicit_aspect_labels is not None and implicit_opinion_labels is not None:
            device = input_ids.device
            
            # --- NOVITÀ 1: LOSS DEL CRF ---
            # Il CRF non tollera le label "-100" (usate spesso per il padding in HF).
            # Creiamo una maschera valida e sostituiamo i -100 con 0 per sicurezza.
            valid_mask = (labels >= 0) & attention_mask.bool()
            safe_labels = torch.where(labels >= 0, labels, torch.zeros_like(labels))
            
            # Calcoliamo la loss negativa della log-likelihood del CRF
            loss_token = -self.crf(emissions, safe_labels, mask=valid_mask, reduction='mean')
            
            # --- NOVITÀ 2: Pesi per gli Impliciti (Intatti!) ---
            # Classe 0 (Esplicito) = 1.0. Classe 1 (Implicito) = 5.0.
            implicit_weights = torch.tensor([1.0, 5.0], device=device)
            loss_fct_implicit = nn.CrossEntropyLoss(weight=implicit_weights)
            
            loss_asp = loss_fct_implicit(imp_asp_logits, implicit_aspect_labels)
            loss_opi = loss_fct_implicit(imp_opi_logits, implicit_opinion_labels)
            
            # --- NOVITÀ 3: Moltiplicatori della Loss Multi-Task ---
            loss = loss_token + (1.5 * loss_asp) + (2.0 * loss_opi)
            
            
        # --- FASE DI INFERENZA (Decodifica) ---
        # Usiamo l'algoritmo di Viterbi per trovare la sequenza grammaticalmente perfetta!
        mask_crf = attention_mask.bool()
        token_preds = self.crf.decode(emissions, mask=mask_crf)
        
        # Poiché il CRF restituisce liste di lunghezza variabile (taglia via il padding),
        # le ri-paddiamo con zeri per restituire un tensore uniforme e non rompere il tuo test.
        batch_size = input_ids.shape[0]
        max_seq_length = input_ids.shape[1]
        padded_preds = []
        
        for i in range(batch_size):
            pred = token_preds[i]
            pad_len = max_seq_length - len(pred)
            padded_preds.append(pred + [0] * pad_len)
            
        # Restituiamo il tensore finale
        token_preds_tensor = torch.tensor(padded_preds, device=input_ids.device)
            
        return {
            "loss": loss, 
            "token_logits": token_preds_tensor,
            "imp_asp_logits": imp_asp_logits, 
            "imp_opi_logits": imp_opi_logits
        }

#FASE 2: Classificazione di Categoria e Sentimento (ACOS)
class ModernBertACOSClassifier(nn.Module):
    def __init__(self, path_to_best_model, num_categories):
        super(ModernBertACOSClassifier, self).__init__()
        
        # Carichiamo il corpo dal modello Step 1
        self.modernbert = AutoModel.from_pretrained(path_to_best_model)
        hidden_size = self.modernbert.config.hidden_size # 768
        
        # Le 121 teste ORA PRENDONO 768 (non più 1536)
        self.heads = nn.ModuleList([
            nn.Linear(hidden_size, 4) for _ in range(num_categories)
        ])
        
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask): # Span rimossi dai parametri!
        outputs = self.modernbert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Prendiamo semplicemente il token [CLS] dell'intera sequenza Cross-Encoder
        cls_output = outputs.last_hidden_state[:, 0, :] 
        cls_output = self.dropout(cls_output)

        # Passiamo il vettore nelle teste lineari
        logits = [head(cls_output) for head in self.heads]
        
        return torch.stack(logits, dim=1)
    
