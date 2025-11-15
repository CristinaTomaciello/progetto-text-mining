# Progetto-text-mining
Generazione di "meta-recensioni" (riassunti collettivi) che siano quantificate, specifiche per aspetto e basate su evidenze tratte da molte recensioni.
-Obiettivo: Creare un riassunto che non sia solo la media del sentiment di un prodotto/servizio, ma che riassuma i PRO e i CONTRO discussi da tutti gli utenti, idealmente suddivisi per aspetto (es. cibo, posizione, servizio).
-Limite degli Approcci Esistenti (es. dataset SPACE): Il dataset SPACE, pur essendo un benchmark (punto di riferimento), fornisce riassunti umani che sono spesso troppo vaghi, non quantificati (non dicono "l'83% degli utenti pensa...") e senza riferimenti alle recensioni originali.
-La Visione Finale: Generare riassunti altamente informativi che dicano cose come: "l'83% degli utenti afferma che la pizza margherita è poco cotta."

Il professore suggerisce di non concentrarsi subito sulla generazione del riassunto, ma sull'Estrazione di Informazioni (IE) più precise e utili, per poi usarle per generare un riassunto di alta qualità.
Il progetto si articola in tre fasi principali:

Fase 1: Estrazione di Informazioni Dettagliate (ACOS)
Il cuore del progetto è l'implementazione del modello descritto nel paper ACL 2021 (ACOS).
Task: Estrazione di Quadruple (Quadruple Extraction).
Dato di Input: Una singola frase/recensione.
Dato di Output (la Quadrupla): Un set di quattro elementi che catturano l'opinione in modo completo:
Item/Entità menzionata (es. pizza margherita).
Categoria di Aspetto (es. Food).
Opinione specifica (es. poco cotta).
Classe di Sentiment (es. Negative).
Azione Richiesta: Addestrare un modello all'avanguardia (come una versione moderna di BERT/ModernBERT) sui dataset forniti (Restaurant-ACOS, Laptop-ACOS) per eseguire questa estrazione. L'obiettivo è superare, o almeno replicare, lo State-of-the-Art del 2021.

Fase 2: Applicazione e Aggregazione Statistica
Una volta addestrato il modello ACOS, lo si applica per raccogliere informazioni da un grande set di dati:
Applicazione: Utilizzare il modello ACOS per estrarre quadruple da tutte le recensioni del dataset SPACE (quelle relative agli hotel).
Aggregazione: Dopo aver estratto migliaia di quadruple (es. [service, fast, Positive] o [bed, room, uncomfortable, Negative]), devi aggregarle automaticamente per ottenere statistiche significative.
Esempio di output aggregato: "Su 1000 quadruple relative all'aspetto 'Food', 830 hanno come opinione 'poco cotta' e sentiment 'Negative'."
Obiettivo: Ottenere opinioni informative e quantificate.

Fase 3: Generazione del Riassunto e Valutazione
L'ultima fase consiste nel produrre il riassunto finale e valutarne la qualità.
Generazione del Riassunto (Verbalizzazione): Le statistiche aggregate (Fase 2) devono essere passate a un Modello Linguistico di Grande Dimensione (LLM).
Metodo: Probabilmente tramite Prompting (chiedendo all'LLM di trasformare i dati statistici in un testo fluente e coerente, come "Il X% degli utenti afferma che...").

Risultato Atteso: Un riassunto con claim quantificati e riferenziabili (anche se i riferimenti visivi non saranno implementati, l'informazione è lì).
Valutazione (Benchmarking):
Qualità del Riassunto: Confrontare la qualità del riassunto generato con l'LLM con gli attuali gold standard (riassunti umani) del dataset SPACE.

Metodo di Valutazione: Utilizzare l'approccio "LLM-as-a-judge" (un altro LLM valuta la qualità e la precisione dei riassunti generati).

RIFERIMENTI:
GITHUB ACOS: https://github.com/NUSTM/ACOS/tree/main

PAPER ACOS: https://aclanthology.org/2021.acl-long.29/

PAPER SPACE: https://arxiv.org/pdf/2012.04443

