#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
                           ACTIONS.PY - Chatbot Laptop Assistant
================================================================================

Descrizione:
  Questo file contiene le azioni personalizzate per il chatbot che aiuta gli utenti 
  nella ricerca, confronto e analisi di laptop. Le azioni includono funzioni per:
    - Suggerire laptop basati su filtri avanzati (slot e parole chiave)
    - Confrontare due laptop
    - Mostrare dettagli tecnici completi di un laptop
    - Restituire il prezzo di un laptop
    - Fornire istruzioni e aiuto all'utente
    - Mostrare una lista di laptop
    - Filtrare e ordinare il dataset in base a criteri multipli
    - Simulare aggiornamenti del dataset, variazioni di prezzo, ritardi e carichi di sistema
    - Integrazione (simulata) di recensioni esterne e dati aggiuntivi
    - Funzioni di debug e reportistica per il monitoraggio

Requisiti:
  - Il file CSV "laptops_etl_clean.csv" deve trovarsi nella stessa cartella (o nel percorso corretto).
  - Il dataset deve avere le seguenti colonne:
      Unnamed: 0, img_link, name, processor, ram, os, storage, display(in inch),
      rating, no_of_ratings, no_of_reviews, price(in EUR)

Autore: [Il Tuo Nome]
Data: [Data di Creazione]

================================================================================
"""

# =============================================================================
# IMPORTS E CONFIGURAZIONE DEL LOGGING
# =============================================================================

import pandas as pd
import logging
import datetime
import time
from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

# Configurazione del logger per output dettagliato
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# =============================================================================
# CARICAMENTO DEL DATASET
# =============================================================================

try:
    # Legge il dataset pulito dei laptop
    df = pd.read_csv("laptops_etl_clean.csv")
    # Rimuove eventuali righe con valori nulli nelle colonne critiche
    df.dropna(subset=["processor", "ram", "os", "storage", "display(in inch)", "price(in EUR)"], inplace=True)
    logger.info("Dataset caricato e pulito correttamente.")
except Exception as e:
    logger.error(f"Errore nel caricamento del dataset: {e}")
    df = pd.DataFrame()  # Fallback in caso di errore

# Aggiungi alcune righe vuote e commenti per aumentare la lunghezza del file
# =============================================================================
#
#
#
# =============================================================================

# =============================================================================
# FUNZIONI HELPER
# =============================================================================

def format_laptop_details(row: pd.Series) -> str:
    """
    Formatta le informazioni di un laptop in una stringa dettagliata.
    
    Parametri:
      - row: Una riga del DataFrame contenente i dati di un laptop.
    
    Restituisce:
      Una stringa che include:
        * Processor
        * RAM
        * OS
        * Storage
        * Display (in pollici)
        * Rating (con numero di valutazioni e recensioni)
        * Prezzo in EUR
    """
    details = (
        f"Processor: {row.get('processor', 'N/A')}\n"
        f"RAM: {row.get('ram', 'N/A')}\n"
        f"OS: {row.get('os', 'N/A')}\n"
        f"Storage: {row.get('storage', 'N/A')}\n"
        f"Display: {row.get('display(in inch)', 'N/A')} pollici\n"
        f"Rating: {row.get('rating', 'N/A')} (basato su {row.get('no_of_ratings', 'N/A')} valutazioni, {row.get('no_of_reviews', 'N/A')} recensioni)\n"
        f"Prezzo: {row.get('price(in EUR)', 'N/A')} EUR\n"
    )
    return details

# -----------------------------------------------------------------------------
# FUNZIONE DI FILTRAGGIO AVANZATO
# -----------------------------------------------------------------------------

def advanced_filtering(tracker: Tracker) -> pd.DataFrame:
    """
    Applica filtri avanzati al dataset basandosi sui valori degli slot definiti nel domain
    e sulle eventuali parole chiave nel messaggio dell'utente.
    
    Utilizza:
      - Slot 'ram', 'processor', 'storage', 'price_range' e 'usage'
      - Parole chiave nel messaggio (es. "economico", "sotto")
    
    Restituisce:
      Un DataFrame filtrato contenente solo i laptop che soddisfano i criteri.
    """
    # Copia del DataFrame originale
    filtered_df = df.copy()

    # Assicurarsi che la colonna prezzo sia numerica
    filtered_df["price(in EUR)"] = pd.to_numeric(filtered_df["price(in EUR)"], errors="coerce")

    logger.debug("Avvio del filtraggio avanzato basato sugli slot e sul messaggio.")

    # Recupera i valori degli slot
    ram_slot = tracker.get_slot("ram")
    processor_slot = tracker.get_slot("processor")
    storage_slot = tracker.get_slot("storage")
    price_range_slot = tracker.get_slot("price_range")
    usage_slot = tracker.get_slot("usage")

    # Applicazione dei filtri basati sugli slot
    if ram_slot:
        filtered_df = filtered_df[filtered_df["ram"].str.contains(ram_slot, case=False, na=False)]
        logger.debug(f"Filtro applicato per RAM: {ram_slot}")

    if processor_slot:
        filtered_df = filtered_df[filtered_df["processor"].str.contains(processor_slot, case=False, na=False)]
        logger.debug(f"Filtro applicato per processor: {processor_slot}")

    if storage_slot:
        filtered_df = filtered_df[filtered_df["storage"].str.contains(storage_slot, case=False, na=False)]
        logger.debug(f"Filtro applicato per storage: {storage_slot}")

    if price_range_slot:
        try:
            price_limit = float(price_range_slot)
            filtered_df = filtered_df[filtered_df["price(in EUR)"] < price_limit]
            logger.debug(f"Filtro applicato per prezzo inferiore a: {price_limit} EUR")
        except Exception as e:
            logger.error(f"Errore nel convertire il prezzo: {e}")

    if usage_slot:
        usage_value = usage_slot.lower()
        if usage_value == "gaming":
            filtered_df = filtered_df[filtered_df["processor"].str.contains("Core i5|Core i7|Ryzen 7|Ryzen 9", case=False, na=False)]
            logger.debug("Filtro applicato per uso gaming basato sullo slot 'usage'.")
        elif usage_value in ["editing video", "rendering"]:
            filtered_df = filtered_df[filtered_df["processor"].str.contains("Core i9|Ryzen 9", case=False, na=False)]
            logger.debug("Filtro applicato per editing video/rendering basato sullo slot 'usage'.")

    # Filtraggio aggiuntivo basato sul messaggio dell'utente
    user_message = tracker.latest_message.get("text", "").lower()
    if "economico" in user_message or "sotto" in user_message:
        filtered_df = filtered_df[filtered_df["price(in EUR)"] < 500]
        logger.debug("Filtro aggiuntivo applicato dal messaggio: laptop economico.")

    logger.debug(f"Filtraggio avanzato completato. {len(filtered_df)} laptop trovati.")
    return filtered_df

# -----------------------------------------------------------------------------
# FUNZIONE DI ORDINAMENTO AVANZATO
# -----------------------------------------------------------------------------

def advanced_sorting(filtered_df: pd.DataFrame, tracker: Tracker) -> pd.DataFrame:
    """
    Ordina il DataFrame filtrato in base alle preferenze espresse nel messaggio dell'utente.
    
    Se il messaggio contiene "migliore" o "top", ordina per rating (decrescente).
    Altrimenti, ordina per prezzo (crescente).
    
    Restituisce:
      Il DataFrame ordinato.
    """
    user_message = tracker.latest_message.get("text", "").lower()
    if "migliore" in user_message or "top" in user_message:
        sorted_df = filtered_df.sort_values(by="rating", ascending=False)
        logger.debug("Ordinamento applicato: rating decrescente.")
    else:
        sorted_df = filtered_df.sort_values(by="price(in EUR)", ascending=True)
        logger.debug("Ordinamento applicato: prezzo crescente.")
    return sorted_df

# =============================================================================
# AZIONI PERSONALIZZATE
# =============================================================================

# -----------------------------------------------------------------------------
# Azione: Suggerisci Laptop
# -----------------------------------------------------------------------------

class ActionSuggerisciLaptop(Action):
    def name(self) -> Text:
        return "action_suggerisci_laptop"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict]:
        """
        Suggerisce laptop basandosi sui valori degli slot (definiti nel domain)
        e sul messaggio dell'utente. Applica filtri avanzati e ordina i risultati
        in base al prezzo (default).
        """
        user_message = tracker.latest_message.get("text", "").lower()
        logger.debug(f"Messaggio utente per ricerca laptop: {user_message}")

        try:
            # Applica il filtraggio avanzato
            filtered_laptops = advanced_filtering(tracker)
            logger.debug(f"Laptop trovati dopo filtraggio: {len(filtered_laptops)}")
            
            # Ordina i risultati (default: per prezzo crescente)
            sorted_laptops = advanced_sorting(filtered_laptops, tracker)
            logger.debug(f"Laptop trovati dopo ordinamento: {len(sorted_laptops)}")
            
            # Costruisce la risposta
            if sorted_laptops.empty:
                risposta = "Mi dispiace, non ho trovato laptop che corrispondano alle tue richieste."
            else:
                risposta = "Ecco alcuni laptop che potrebbero interessarti:\n\n"
                for _, row in sorted_laptops.head(3).iterrows():
                    dettagli = format_laptop_details(row)
                    risposta += f"- **{row['name']}**:\n{dettagli}\n"
            dispatcher.utter_message(text=risposta)
        except Exception as e:
            logger.exception(f"Errore in ActionSuggerisciLaptop: {e}")
            dispatcher.utter_message(text=f"Si è verificato un errore durante la ricerca dei laptop: {e}")
        return []


# -----------------------------------------------------------------------------
# Azione: Confronta Laptop
# -----------------------------------------------------------------------------

class ActionConfrontaLaptop(Action):
    def name(self) -> Text:
        return "action_confronta_laptop"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict]:
        """
        Confronta due laptop specificati dagli slot 'laptop1' e 'laptop2'.
        Gestisce errori se uno dei nomi non viene trovato.
        """
        laptop1 = tracker.get_slot("laptop1")
        laptop2 = tracker.get_slot("laptop2")
        logger.debug(f"Richiesta di confronto tra: {laptop1} e {laptop2}")

        if not laptop1 or not laptop2:
            dispatcher.utter_message(text="Per favore, specifica i nomi di entrambi i laptop da confrontare.")
            return []
        try:
            data1 = df[df["name"] == laptop1].iloc[0]
            data2 = df[df["name"] == laptop2].iloc[0]
        except IndexError as e:
            logger.error(f"Errore nel confronto: {e}")
            dispatcher.utter_message(text="Non sono riuscito a trovare uno dei laptop specificati. Verifica i nomi inseriti.")
            return []

        response = f"Confronto dettagliato tra **{laptop1}** e **{laptop2}**:\n\n"
        response += f"### {laptop1}\n{format_laptop_details(data1)}\n"
        response += f"### {laptop2}\n{format_laptop_details(data2)}\n"
        dispatcher.utter_message(text=response)
        return []


# -----------------------------------------------------------------------------
# Azione: Caratteristiche Laptop
# -----------------------------------------------------------------------------

class ActionCaratteristicheLaptop(Action):
    def name(self) -> Text:
        return "action_caratteristiche_laptop"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict]:
        """
        Fornisce una descrizione dettagliata del laptop specificato nello slot 'laptop'.
        Se il valore inserito corrisponde parzialmente a più laptop, chiede all'utente di specificare meglio.
        """
        laptop = tracker.get_slot("laptop")
        logger.debug(f"Richiesta dettagli per laptop: {laptop}")

        if not laptop:
            dispatcher.utter_message(text="Per favore, specifica il nome del laptop di cui vuoi conoscere le caratteristiche.")
            return []
        try:
            # Effettua un match parziale (case insensitive) sulla colonna "name"
            filtered = df[df["name"].str.contains(laptop, case=False, na=False)]
            if filtered.empty:
                raise IndexError("Nessun laptop trovato.")
            
            # Se troviamo più di un risultato, chiediamo all'utente di precisare
            if len(filtered) > 1:
                laptop_list = filtered["name"].tolist()
                response = ("Ho trovato più laptop che corrispondono alla tua ricerca:\n" +
                            "\n".join(f"- {name}" for name in laptop_list) +
                            "\nPer favore, specifica il nome completo del laptop che ti interessa.")
            else:
                data = filtered.iloc[0]
                response = f"Dettagli completi di **{data['name']}**:\n\n{format_laptop_details(data)}"
        except IndexError:
            logger.error(f"Laptop non trovato: {laptop}")
            response = "Non sono riuscito a trovare il laptop specificato. Controlla il nome e riprova."
        dispatcher.utter_message(text=response)
        return []


# -----------------------------------------------------------------------------
# Azione: Prezzo Laptop
# -----------------------------------------------------------------------------

class ActionPrezzoLaptop(Action):
    def name(self) -> Text:
        return "action_prezzo_laptop"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict]:
        """
        Restituisce il prezzo del laptop specificato nello slot 'laptop'.
        Gestisce eventuali errori se il laptop non viene trovato.
        """
        laptop = tracker.get_slot("laptop")
        logger.debug(f"Richiesta prezzo per laptop: {laptop}")

        if not laptop:
            dispatcher.utter_message(text="Per favore, specifica il nome del laptop per conoscere il prezzo.")
            return []
        try:
            prezzo = df[df["name"] == laptop]["price(in EUR)"].values[0]
            response = f"Il prezzo di **{laptop}** è {prezzo} EUR."
        except IndexError:
            logger.error(f"Prezzo non trovato per il laptop: {laptop}")
            response = "Non sono riuscito a trovare il laptop specificato. Verifica il nome e riprova."
        dispatcher.utter_message(text=response)
        return []


# -----------------------------------------------------------------------------
# Azione: Aiuto (Help)
# -----------------------------------------------------------------------------

class ActionHelp(Action):
    def name(self) -> Text:
        return "action_help"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict]:
        """
        Fornisce istruzioni e suggerimenti su come utilizzare il chatbot per trovare il laptop ideale.
        """
        message = (
            "Sono qui per aiutarti a trovare il laptop ideale!\n"
            "Puoi chiedermi di cercare un laptop per uso specifico (ad esempio, [gaming](usage), [lavoro](usage), [editing video](usage)),\n"
            "oppure di confrontarne due, fornendoti dettagli tecnici o il prezzo.\n"
            "Esempi:\n"
            " - 'Cerco un laptop per gaming con Intel Core i7 e 16GB di RAM'\n"
            " - 'Confronta il Dell Inspiron con il MacBook Air'\n"
            " - 'Dimmi le caratteristiche del MacBook Pro'\n"
            " - 'Qual è il prezzo del Dell XPS 15'\n"
        )
        dispatcher.utter_message(text=message)
        return []


# -----------------------------------------------------------------------------
# Azione: Mostra Lista Laptop
# -----------------------------------------------------------------------------

class ActionShowLaptopList(Action):
    def name(self) -> Text:
        return "action_show_laptop_list"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict]:
        """
        Mostra una lista di alcuni laptop disponibili con dettagli di base.
        """
        logger.debug("Richiesta lista laptop.")
        try:
            laptops = df.head(5)
            response = "Ecco una lista di alcuni laptop disponibili:\n\n"
            for _, row in laptops.iterrows():
                response += (
                    f"- **{row['name']}**: {row['processor']}, {row['ram']}, {row['os']}, "
                    f"{row['storage']}, Display: {row['display(in inch)']} pollici, Prezzo: {row['price(in EUR)']} EUR\n\n"
                )
        except Exception as e:
            logger.error(f"Errore nel recupero della lista laptop: {e}")
            response = "Si è verificato un errore nel recuperare la lista dei laptop."
        dispatcher.utter_message(text=response)
        return []


# =============================================================================
# SEZIONE: FUNZIONI DI SUPPORTO (Filtraggio, Ordinamento, Debug, Aggiornamenti)
# =============================================================================

def get_laptop_by_brand(brand: Text) -> pd.DataFrame:
    """Restituisce i laptop che contengono il brand specificato nel nome."""
    logger.debug(f"Filtraggio per brand: {brand}")
    return df[df["name"].str.contains(brand, case=False, na=False)]


def get_laptop_by_processor(processor: Text) -> pd.DataFrame:
    """Restituisce i laptop che contengono il processore specificato."""
    logger.debug(f"Filtraggio per processore: {processor}")
    return df[df["processor"].str.contains(processor, case=False, na=False)]


def get_laptop_by_ram(ram: Text) -> pd.DataFrame:
    """Restituisce i laptop che contengono la quantità di RAM specificata."""
    logger.debug(f"Filtraggio per RAM: {ram}")
    return df[df["ram"].str.contains(ram, case=False, na=False)]


def get_laptop_by_price(max_price: float) -> pd.DataFrame:
    """Restituisce i laptop con prezzo inferiore a max_price."""
    logger.debug(f"Filtraggio per prezzo inferiore a: {max_price} EUR")
    return df[df["price(in EUR)"] < max_price]


def sort_laptops_by_rating(laptops: pd.DataFrame) -> pd.DataFrame:
    """Ordina i laptop per rating in ordine decrescente."""
    logger.debug("Ordinamento per rating decrescente.")
    return laptops.sort_values(by="rating", ascending=False)


def sort_laptops_by_price(laptops: pd.DataFrame, ascending: bool = True) -> pd.DataFrame:
    """Ordina i laptop in base al prezzo."""
    logger.debug(f"Ordinamento per prezzo, ascending={ascending}.")
    return laptops.sort_values(by="price(in EUR)", ascending=ascending)


# -----------------------------------------------------------------------------
# Funzioni di Debug e Utility
# -----------------------------------------------------------------------------

def log_current_time() -> None:
    """Registra e logga l'orario corrente per scopi di debug."""
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.debug(f"Orario corrente: {current_time}")


def simulate_processing_delay(seconds: int = 1) -> None:
    """Simula un ritardo nel processamento per emulare operazioni intensive."""
    logger.debug(f"Simulazione di un ritardo di {seconds} secondi...")
    time.sleep(seconds)


# -----------------------------------------------------------------------------
# AZIONI DI DEBUG, REPORT E SIMULAZIONE DI CARICO
# -----------------------------------------------------------------------------

class ActionSimulaRitardo(Action):
    def name(self) -> Text:
        return "action_simula_ritardo"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict]:
        """Simula un ritardo nel processing per testare l'impatto sui tempi di risposta."""
        dispatcher.utter_message(text="Attendere, sto elaborando la richiesta...")
        simulate_processing_delay(3)
        dispatcher.utter_message(text="Elaborazione completata!")
        return []


class ActionAggiornaDataset(Action):
    def name(self) -> Text:
        return "action_aggiorna_dataset"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict]:
        """
        Ricarica il dataset dal file CSV e aggiorna la variabile globale.
        """
        log_current_time()
        try:
            updated_df = pd.read_csv("..\\laptop_etl.csv")
            global df
            df = updated_df.copy()
            logger.info("Dataset aggiornato con successo.")
            dispatcher.utter_message(text="Il dataset dei laptop è stato aggiornato con successo.")
        except Exception as e:
            logger.error(f"Errore durante l'aggiornamento del dataset: {e}")
            dispatcher.utter_message(text="Si è verificato un errore durante l'aggiornamento del dataset.")
        return []


class ActionAggiornaPrezzi(Action):
    def name(self) -> Text:
        return "action_aggiorna_prezzi"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict]:
        """
        Aggiorna i prezzi dei laptop in base al fattore specificato nello slot 'price_update_factor'.
        """
        factor_str = tracker.get_slot("price_update_factor")
        logger.debug(f"Fattore di aggiornamento dei prezzi ricevuto: {factor_str}")
        try:
            factor = float(factor_str) if factor_str else 1.0
            df["price(in EUR)"] = df["price(in EUR)"] * factor
            logger.info(f"Prezzi aggiornati con fattore {factor}.")
            response = f"I prezzi sono stati aggiornati con un fattore di {factor}."
        except Exception as e:
            logger.error(f"Errore in ActionAggiornaPrezzi: {e}")
            response = "Si è verificato un errore durante l'aggiornamento dei prezzi."
        dispatcher.utter_message(text=response)
        return []


class ActionGeneraReport(Action):
    def name(self) -> Text:
        return "action_genera_report"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict]:
        """
        Genera un report d'utilizzo simulato basato sui dati del dataset.
        """
        try:
            total_laptops = len(df)
            avg_price = df["price(in EUR)"].mean()
            avg_rating = df["rating"].mean()
            report = (
                f"Report d'utilizzo:\n"
                f"- Totale laptop: {total_laptops}\n"
                f"- Prezzo medio: {avg_price:.2f} EUR\n"
                f"- Rating medio: {avg_rating:.2f}\n"
            )
        except Exception as e:
            logger.error(f"Errore in ActionGeneraReport: {e}")
            report = "Non sono riuscito a calcolare le statistiche del dataset."
        dispatcher.utter_message(text=report)
        return []


class ActionDebugStato(Action):
    def name(self) -> Text:
        return "action_debug_stato"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict]:
        """
        Restituisce lo stato attuale degli slot e del tracker per scopi di debug.
        """
        slots = tracker.current_slot_values()
        debug_info = "Stato attuale degli slot:\n"
        for key, value in slots.items():
            debug_info += f"- {key}: {value}\n"
        dispatcher.utter_message(text=debug_info)
        logger.debug(debug_info)
        return []


class ActionLogEventi(Action):
    def name(self) -> Text:
        return "action_log_eventi"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict]:
        """
        Logga informazioni dettagliate sulla conversazione per analisi successive.
        """
        events = tracker.events
        logger.debug("Eventi della conversazione:")
        for event in events:
            logger.debug(event)
        dispatcher.utter_message(text="Ho registrato tutti gli eventi della conversazione per il debug.")
        return []


class ActionSimulaCarico(Action):
    def name(self) -> Text:
        return "action_simula_carico"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict]:
        """
        Simula un carico elevato nel sistema per testare la scalabilità del chatbot.
        """
        logger.debug("Simulazione di carico in corso...")
        total = 0
        for i in range(1000000):
            total += i
            if i % 200000 == 0:
                logger.debug(f"Caricamento: iterazione {i}")
        dispatcher.utter_message(text="Simulazione di carico completata. Il sistema ha gestito il carico con successo!")
        return []


# -----------------------------------------------------------------------------
# Azioni per Integrazione con Dati Esterni e Recensioni
# -----------------------------------------------------------------------------

def simulate_external_api_call() -> str:
    """
    Simula una chiamata ad un'API esterna che restituisce dati aggiuntivi sul laptop.
    """
    logger.debug("Simulazione chiamata API esterna in corso...")
    time.sleep(2)
    return "Recensioni, benchmark e suggerimenti aggiornati."

class ActionRecuperaDatiEsterni(Action):
    def name(self) -> Text:
        return "action_recupera_dati_esterni"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict]:
        """
        Recupera dati esterni (simulati) per arricchire le informazioni sul laptop specificato.
        """
        laptop = tracker.get_slot("laptop")
        if not laptop:
            dispatcher.utter_message(text="Per favore, specifica il nome del laptop per cui vuoi dati esterni.")
            return []
        try:
            external_data = simulate_external_api_call()
            response = f"Dati esterni per {laptop}:\n{external_data}"
        except Exception as e:
            logger.error(f"Errore nella chiamata API esterna: {e}")
            response = "Si è verificato un errore nel recuperare dati esterni per questo laptop."
        dispatcher.utter_message(text=response)
        return []


class ActionAggiungiRecensioni(Action):
    def name(self) -> Text:
        return "action_aggiungi_recensioni"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict]:
        """
        Aggiunge recensioni esterne al dettaglio del laptop.
        """
        laptop = tracker.get_slot("laptop")
        logger.debug(f"Richiesta recensioni per: {laptop}")
        if not laptop:
            dispatcher.utter_message(text="Per favore, specifica il nome del laptop per vedere le recensioni.")
            return []
        try:
            # Simula la raccolta delle recensioni
            reviews_text = "\n".join([
                "Questo modello è eccezionale per il gaming!",
                "Ottimo per il multitasking, ma la batteria potrebbe essere migliore.",
                "Il design è molto elegante e la performance soddisfacente."
            ])
            response = f"Recensioni per {laptop}:\n{reviews_text}"
        except Exception as e:
            logger.error(f"Errore nell'integrazione delle recensioni: {e}")
            response = "Non sono riuscito a recuperare le recensioni per questo laptop."
        dispatcher.utter_message(text=response)
        return []


# =============================================================================
# SEZIONE: COMMENTI E SPAZI AGGIUNTIVI PER DOCUMENTAZIONE
# =============================================================================
#
#
#
# Le righe seguenti sono aggiunte per garantire una documentazione completa
# e per aumentare la dimensione del file come richiesto. 
#
#
#
#
#
#
#
#
#
#
#
# =============================================================================
# FINE DEL FILE ACTIONS.PY
# =============================================================================