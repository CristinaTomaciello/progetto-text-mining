import torch
import torch.nn as nn
from transformers import AutoModel
from torchcrf import CRF        
def get_spans(tags, b_tag, i_tag):
    spans = []
    start = -1
    for i, tag in enumerate(tags):
        if tag == b_tag:
            if start != -1: spans.append((start, i))
            start = i
        elif tag == i_tag and start != -1: continue
        else:
            if start != -1:
                spans.append((start, i))
                start = -1
    if start != -1: spans.append((start, len(tags)))
    return spans

def predict_quadruples_e2e(text, model_1, model_2, tokenizer, cat_list, device, id2label, best_threshold):
    words = text.split()
    inputs = tokenizer(words, is_split_into_words=True, return_tensors="pt", truncation=True, max_length=128, padding='max_length').to(device)
    
    # --- FASE 1 ---
    with torch.no_grad():
        with torch.amp.autocast(device_type=device.type):
            out1 = model_1(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        
    token_preds = out1['token_logits'][0].cpu().numpy()
    imp_asp = torch.argmax(out1['imp_asp_logits'], dim=-1)[0].item()
    imp_opi = torch.argmax(out1['imp_opi_logits'], dim=-1)[0].item()
    
    word_ids = inputs.word_ids()
    word_tags = ["O"] * len(words)
    
    for idx, w_id in enumerate(word_ids):
        if w_id is not None and w_id < len(words) and word_tags[w_id] == "O":
            word_tags[w_id] = id2label[token_preds[idx]]
                
    asp_spans = get_spans(word_tags, "B-ASP", "I-ASP")
    opi_spans = get_spans(word_tags, "B-OPI", "I-OPI")
    
    if imp_asp == 1 or len(asp_spans) == 0: asp_spans.append((-1, -1))
    if imp_opi == 1 or len(opi_spans) == 0: opi_spans.append((-1, -1))
    
    asp_spans = list(set(asp_spans))
    opi_spans = list(set(opi_spans))
    quadruples = []
    
    # --- FASE 2 ---
    for a in asp_spans:
        for o in opi_spans:
            
            # Estraiamo le stringhe
            asp_str = " ".join(words[a[0]:a[1]]) if a != (-1, -1) else "null"
            opi_str = " ".join(words[o[0]:o[1]]) if o != (-1, -1) else "null"
            cross_text = f"aspect: {asp_str} opinion: {opi_str}"
            
            # Re-Tokenizziamo al volo per il Cross-Encoder
            pair_inputs = tokenizer(
                text, cross_text, 
                return_tensors="pt", truncation=True, 
                max_length=128, padding='max_length'
            ).to(device)
            
            with torch.no_grad():
                with torch.amp.autocast(device_type=device.type):
                    out2 = model_2(
                        input_ids=pair_inputs['input_ids'], 
                        attention_mask=pair_inputs['attention_mask']
                    )
            
            logits = out2['logits'] if isinstance(out2, dict) else out2 
            probs = torch.softmax(logits[0], dim=-1) 
            
            for cat_idx, prob_dist in enumerate(probs):
                
                prob_invalido = prob_dist[3].item()
                
                # Se il modello è sicuro oltre il best_threshold che sia spazzatura, scartiamo!
                if prob_invalido > best_threshold:
                    continue
                
                # Altrimenti, ha passato il controllo! 
                # Prendiamo il sentimento più alto tra i primi 3 (Pos, Neg, Neu) ignorando l'Invalido
                best_sentiment = torch.argmax(prob_dist[:3]).item()
                
                quadruples.append({
                    'aspect_span': a,
                    'opinion_span': o,
                    'category': cat_list[cat_idx],
                    'sentiment': best_sentiment
                })
                    
    return quadruples

# VERSIONE CORRETTA per il dataset SPACE CHE USA LE STRINGHE E NON LE COORDINATE!
# La logica è la stessa, ma invece di restituire le coordinate degli span, restituiamo direttamente le stringhe corrispondenti.
def predict_quadruples_space(text, model_1, model_2, tokenizer, cat_list, device, id2label, best_threshold):
    words = text.split()
    inputs = tokenizer(words, is_split_into_words=True, return_tensors="pt", truncation=True, max_length=128, padding='max_length').to(device)
    
    # --- FASE 1 ---
    with torch.no_grad():
        # Aggiunta sicurezza per l'autocast se usi MPS (Mac)
        if device.type == 'cuda' or device.type == 'mps':
            with torch.amp.autocast(device_type=device.type):
                out1 = model_1(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        else:
            out1 = model_1(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        
    token_preds = out1['token_logits'][0].cpu().numpy()
    imp_asp = torch.argmax(out1['imp_asp_logits'], dim=-1)[0].item()
    imp_opi = torch.argmax(out1['imp_opi_logits'], dim=-1)[0].item()
    
    word_ids = inputs.word_ids()
    word_tags = ["O"] * len(words)
    
    for idx, w_id in enumerate(word_ids):
        if w_id is not None and w_id < len(words) and word_tags[w_id] == "O":
            word_tags[w_id] = id2label[token_preds[idx]]
                
    asp_spans = get_spans(word_tags, "B-ASP", "I-ASP")
    opi_spans = get_spans(word_tags, "B-OPI", "I-OPI")
    
    if imp_asp == 1 or len(asp_spans) == 0: asp_spans.append((-1, -1))
    if imp_opi == 1 or len(opi_spans) == 0: opi_spans.append((-1, -1))
    
    asp_spans = list(set(asp_spans))
    opi_spans = list(set(opi_spans))
    quadruples = []
    
    # --- FASE 2 ---
    for a in asp_spans:
        for o in opi_spans:
            
            # Estraiamo le stringhe
            asp_str = " ".join(words[a[0]:a[1]]) if a != (-1, -1) else "null"
            opi_str = " ".join(words[o[0]:o[1]]) if o != (-1, -1) else "null"
            cross_text = f"aspect: {asp_str} opinion: {opi_str}"
            
            # Re-Tokenizziamo al volo per il Cross-Encoder
            pair_inputs = tokenizer(
                text, cross_text, 
                return_tensors="pt", truncation=True, 
                max_length=128, padding='max_length'
            ).to(device)
            
            with torch.no_grad():
                if device.type == 'cuda' or device.type == 'mps':
                    with torch.amp.autocast(device_type=device.type):
                        out2 = model_2(
                            input_ids=pair_inputs['input_ids'], 
                            attention_mask=pair_inputs['attention_mask']
                        )
                else:
                    out2 = model_2(
                        input_ids=pair_inputs['input_ids'], 
                        attention_mask=pair_inputs['attention_mask']
                    )
            
            logits = out2['logits'] if isinstance(out2, dict) else out2 
            probs = torch.softmax(logits[0], dim=-1) 
            
            for cat_idx, prob_dist in enumerate(probs):
                
                prob_invalido = prob_dist[3].item()
                
                # Se il modello è sicuro oltre il best_threshold che sia spazzatura, scartiamo!
                if prob_invalido > best_threshold:
                    continue
                
                # Altrimenti, ha passato il controllo! 
                # Prendiamo il sentimento più alto tra i primi 3 (Pos, Neg, Neu) ignorando l'Invalido
                best_sentiment = torch.argmax(prob_dist[:3]).item()
                
                # ECCO LA PARTE CORRETTA! Usiamo le stringhe e non le coordinate.
                quadruples.append({
                    'aspect_testo': asp_str,  # <--- CORRETTO!
                    'opinion_testo': opi_str, # <--- CORRETTO!
                    'category': cat_list[cat_idx],
                    'sentiment': best_sentiment
                })
                    
    return quadruples