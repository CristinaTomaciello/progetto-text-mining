# ACOS Quadruple Extraction: ModernBERT vs GLiNER 2
Questo repository contiene il codice per il progetto di ricerca sull'estrazione di quadruple ACOS (Aspect-Category-Opinion-Sentiment) basato sull'omonimo task e dataset.

L'obiettivo principale di questo progetto è confrontarsi con i risultati ottenuti nel paper originale degli autori di ACOS, tentando di superare i loro punteggi State-of-the-Art (SOTA) sostituendo le architetture base con modelli del linguaggio più recenti e performanti.

## Modelli Utilizzati e Metodologia
Il progetto esplora due approcci diametralmente opposti per la risoluzione del task:

- ModernBERT (Fine-Tuned con Custom Heads): Sostituisce il BERT-base originale con ModernBERT, che offre una finestra di contesto estesa e rappresentazioni contestuali più ricche.
Il task viene affrontato tramite una pipeline supervisionata composta da due fasi (estrazione span ed estrazione relazioni/classificazione), utilizzando reti neurali custom addestrate specificamente sui dataset ACOS (dominio Laptop e Restaurant).

- GLiNER 2 (Puro Zero-Shot): Un modello unificato per l'estrazione di informazioni guidata da schemi.
Utilizzato in modalità puramente Zero-Shot tramite la funzione extract_json (Structured Data Extraction). Lo scopo è analizzare le reali capacità di comprensione semantica (Entity-Level, Token-Level e Classificazione) dei moderni modelli generalisti senza alcun addestramento specifico sulle regole del dataset.

## Struttura del Repository
I file sono organizzati per dominio (Laptop e Restaurant) e per modello utilizzato:
```text
├── pre-processing-ACOS.ipynb        # Pipeline di allineamento sub-word, etichettatura BIO e parsing dei dataset
├── Laptop-ACOS-ModernBERT.ipynb     # Training e inferenza della pipeline custom su ModernBERT (dominio Laptop)
├── Restaurant-ACOS-ModernBERT.ipynb # Training e inferenza della pipeline custom su ModernBERT (dominio Restaurant)
├── Laptop-ACOS-GLINER2.ipynb        # Valutazione Zero-Shot, metriche seqeval/sklearn e classificazione (dominio Laptop)
├── Restaurant-ACOS-GLINER2.ipynb    # Valutazione Zero-Shot, metriche seqeval/sklearn e classificazione (dominio Restaurant)
├── requirements.txt                 # Dipendenze del progetto
├── data_parsing/                    # Cartella (generata dinamicamente) contenente i .pkl pre-processati per l'inferenza finale (Quadruple Extraction)
├── data_allineati/                  # Cartella per i dataset allineati e etichettati in formato BIO (output del pre-processing)
├── data_coppie/                     # Cartella per i dataset di coppie testo-categoria per l'addestramento della rete custom (output del pre-processing)
├── models/                          # Cartella per i pesi dei modelli ModernBERT addestrati (Fase 1 e Fase 2)
└── risultati_gliner/                # Cartella per il salvataggio delle predizioni Zero-Shot
```

## Setup e Installazione
Per garantire la riproducibilità degli esperimenti, è fortemente consigliato l'uso di un ambiente virtuale (es. venv).

### Clona il repository e posizionati nella cartella:
```bash
git clone <repository_url>
cd <repository_name>
``` 

### Crea e attiva un ambiente virtuale:
```bash
python -m venv venv
source venv/bin/activate  # Su Windows: venv\Scripts\activate
```
### Installa le dipendenze:
```bash
pip install -r requirements.txt
``` 

## Come eseguire il codice

### Step 1: Pre-processing dei Dati
Questo notebook non è necessario da eseguire per poter testare direttamente i modelli, perchè i risultati del pre-processing sono già stati salvati in formato .pkl all'interno del repository. Tuttavia, se desideri eseguire nuovamente il pre-processing o comprendere meglio come sono stati preparati i dati, puoi eseguire il notebook **pre-processing-ACOS.ipynb**. <br>
I dati che servono per poter eseguire i notebook di modernBERT sono:
- **step_1**: i file .pkl generati dal notebook pre-processing-ACOS.ipynb, che contengono i dataset allineati e etichettati in formato BIO, contenuti nella cartella **data_allineati/**.
- **step_2**: i file .pkl generati dal notebook pre-processing-ACOS.ipynb, che contengono le coppie di testo e categorie (label) per l'addestramento della rete custom, contenuti nella cartella **data_coppie/**.

### Step 2: Esecuzione di ModernBERT (Fine-Tuning o Inferenza locale)
- Apri il notebook relativo al dominio che desideri testare (es. Restaurant-ACOS-ModernBERT.ipynb).
Per addestrare il modello da zero: Esegui tutte le celle in sequenza. Il notebook addestrerà le due reti custom e salverà i pesi localmente.

- Per bypassare l'addestramento (Inferenza Rapida): Se non desideri ri-eseguire il training completo, puoi saltare le celle di addestramento. Il codice è predisposto per caricare direttamente i pesi dei modelli delle due fasi (Fase 1 e Fase 2) precedentemente salvati nella cartella locale (es. models/), passando subito alla valutazione tramite Exact Match. La cella per l'inferenza sulla quadrupla completa si trova alla fine del notebook di entrambi i domini. Qui i dati vengono presi dalla cartella **data_parsing/** che è caricata dentro la repository.  <br>

Per poter eseguire l'inferenza rapida, è necessario scaricare i pesi dei modelli addestrati e salvarli localmente nella cartella **models/**. Ecco come fare: 
1. Scarica l'archivio dei modelli: [models.zip](https://liveunibo-my.sharepoint.com/:u:/g/personal/cristina_tomaciello_studio_unibo_it/IQCt16EHmUJ4QZQU7u-AgcTYAeST0XAYBpOlUvutHHB78xw?e=YehUDS)
2. Estrai il contenuto dell'archivio


### Step 3: Valutazione GLiNER 2 (Zero-Shot)
Apri i notebook Laptop-ACOS-GLINER2.ipynb o Restaurant-ACOS-GLINER2.ipynb. Questi notebook non richiedono addestramento: scaricheranno in automatico i pesi del modello base di GLiNER 2 ed eseguiranno l'estrazione strutturata tramite prompt, valutando i risultati attraverso un set di metriche disaccoppiate (Entity-Level F1 con seqeval, Token-Level F1 con sklearn, e Classificazione Astratta).

## Tracciamento degli Esperimenti con Weights & Biases (WandB)
All'interno dei notebook dedicati a ModernBERT, è stata integrata e settata la piattaforma Weights & Biases (wandb).

Perché usiamo WandB?

- Monitoraggio delle Metriche: Permette di tracciare in tempo reale l'andamento della Training Loss e della Validation Loss, essenziale per identificare fenomeni di overfitting durante il fine-tuning della rete custom.

- Logging degli Iperparametri: Registra automaticamente i parametri di addestramento (learning rate, batch size, epoche) per ogni run, rendendo gli esperimenti facilmente riproducibili e comparabili per la stesura dei risultati finali.

- *Nota*: Al primo avvio di una cella di training, ti verrà richiesto di inserire la tua chiave API di WandB (ottenibile gratuitamente registrandosi sul loro sito).

