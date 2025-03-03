import pandas as pd
import logging
from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet

# Configurazione del logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# -------------------- Caricamento del Dataset --------------------
try:
    df = pd.read_csv("laptops_etl_clean.csv")
    logger.info("Dataset caricato correttamente.")
except Exception as e:
    logger.error(f"Errore nel caricamento del dataset: {e}")
    df = pd.DataFrame()

# -------------------- Funzioni Helper --------------------
def format_laptop_details(row: pd.Series) -> str:
    """
    Restituisce una stringa formattata con i dettagli del laptop.
    """
    return (
        f"Processor: {row.get('processor', 'N/A')}\n"
        f"RAM: {row.get('ram', 'N/A')}\n"
        f"OS: {row.get('os', 'N/A')}\n"
        f"Storage: {row.get('storage', 'N/A')}\n"
        f"Display: {row.get('display(in inch)', 'N/A')} pollici\n"
        f"Rating: {row.get('rating', 'N/A')} (basato su {row.get('no_of_ratings', 'N/A')} valutazioni, {row.get('no_of_reviews', 'N/A')} recensioni)\n"
        f"Prezzo: {row.get('price(in EUR)', 'N/A')} EUR\n"
        f"Immagine: {row.get('img_link', 'N/A')}"
    )

def apply_common_filters(tracker: Tracker, df_input: pd.DataFrame) -> pd.DataFrame:
    """
    Applica i filtri comuni basati sugli slot:
    - laptop (nome)
    - ram, processor, storage, os
    - usage (se presente nella colonna 'usage')
    - display (basato su 'display(in inch)')
    """
    df_filtered = df_input.copy()
    
    # Filtra per nome del laptop
    laptop_input = tracker.get_slot("laptop")
    if laptop_input:
        df_filtered = df_filtered[df_filtered["name"].str.contains(laptop_input, case=False, na=False)]
    
    # Filtri per attributi tecnici
    for slot in ["ram", "processor", "storage", "os"]:
        value = tracker.get_slot(slot)
        if value:
            df_filtered = df_filtered[df_filtered[slot].str.contains(value, case=False, na=False)]
    
    # Filtro per usage se presente
    usage_slot = tracker.get_slot("usage")
    if usage_slot and "usage" in df_filtered.columns:
        df_filtered = df_filtered[df_filtered["usage"].str.contains(usage_slot, case=False, na=False)]
    
    # Filtro per display basato sul valore numerico (es. "15.6 pollici")
    display_slot = tracker.get_slot("display")
    if display_slot:
        try:
            display_value = float(display_slot.split()[0])
            df_filtered = df_filtered[df_filtered["display(in inch)"].between(display_value - 0.5, display_value + 0.5)]
        except Exception as e:
            logger.error(f"Errore nel filtraggio per display: {e}")
    return df_filtered

def advanced_filtering(tracker: Tracker) -> pd.DataFrame:
    """
    Applica filtri avanzati basandosi su vari slot e sul messaggio dell'utente.
    Include filtri su ram, processor, storage, brand, prezzo, display, rating e usage.
    """
    df_filtered = df.copy()
    df_filtered["price(in EUR)"] = pd.to_numeric(df_filtered["price(in EUR)"], errors="coerce")
    try:
        df_filtered["display(in inch)"] = df_filtered["display(in inch)"].astype(float)
    except Exception as e:
        logger.error(f"Errore nella conversione di 'display(in inch)': {e}")
    
    logger.debug("Avvio del filtraggio avanzato basato su slot e messaggio.")
    
    # Filtri base
    for slot in ["ram", "processor", "storage"]:
        value = tracker.get_slot(slot)
        if value:
            df_filtered = df_filtered[df_filtered[slot].str.contains(value, case=False, na=False)]
            logger.debug(f"Filtro applicato per {slot}: {value}")
    
    # Filtro per brand (controlla nel nome)
    brand_slot = tracker.get_slot("brand")
    if brand_slot:
        df_filtered = df_filtered[df_filtered["name"].str.contains(brand_slot, case=False, na=False)]
        logger.debug(f"Filtro applicato per brand: {brand_slot}")
    
    # Filtro per prezzo
    price_range_slot = tracker.get_slot("price_range")
    price_condition_slot = tracker.get_slot("price_condition")
    if price_range_slot:
        try:
            price_limit = float(price_range_slot)
            if price_condition_slot == "superiore":
                df_filtered = df_filtered[df_filtered["price(in EUR)"] > price_limit]
                logger.debug(f"Filtro applicato per prezzo superiore a: {price_limit} EUR")
            else:
                df_filtered = df_filtered[df_filtered["price(in EUR)"] < price_limit]
                logger.debug(f"Filtro applicato per prezzo inferiore a: {price_limit} EUR")
        except Exception as e:
            logger.error(f"Errore nella conversione di price_range: {e}")
    
    # Filtro per display
    display_slot = tracker.get_slot("display")
    if display_slot:
        try:
            display_value = float(display_slot.split()[0])
            df_filtered = df_filtered[df_filtered["display(in inch)"].between(display_value - 0.5, display_value + 0.5)]
            logger.debug(f"Filtro applicato per display: {display_value} pollici (±0.5)")
        except Exception as e:
            logger.error(f"Errore nel filtraggio per display: {e}")
    
    # Filtro per rating
    rating_range_slot = tracker.get_slot("rating_range")
    if rating_range_slot:
        try:
            min_rating, max_rating = map(float, rating_range_slot.split("-"))
            df_filtered = df_filtered[(df_filtered["rating"] >= min_rating) & (df_filtered["rating"] <= max_rating)]
            logger.debug(f"Filtro applicato per rating tra {min_rating} e {max_rating}")
        except Exception as e:
            logger.error(f"Errore nel filtraggio per rating_range: {e}")
    
    # Filtro specifico per usage
    usage_slot = tracker.get_slot("usage")
    if usage_slot:
        usage_value = usage_slot.lower()
        if usage_value == "gaming":
            df_filtered = df_filtered[df_filtered["processor"].str.contains("Core i5|Core i7|Ryzen 7|Ryzen 9", case=False, na=False)]
            logger.debug("Filtro applicato per uso gaming.")
        elif usage_value == "lavoro":
            df_filtered = df_filtered[df_filtered["os"].str.contains("Windows", case=False, na=False)]
            logger.debug("Filtro applicato per uso lavoro.")
        elif usage_value == "programmazione":
            df_filtered = df_filtered[df_filtered["processor"].str.contains("Core i5|Core i7|Ryzen 5|Ryzen 7", case=False, na=False)]
            logger.debug("Filtro applicato per uso programmazione.")
        elif usage_value in ["editing video", "rendering"]:
            df_filtered = df_filtered[df_filtered["processor"].str.contains("Core i9|Ryzen 9", case=False, na=False)]
            logger.debug("Filtro applicato per uso editing video/rendering.")
        elif usage_value == "studenti":
            df_filtered = df_filtered[df_filtered["price(in EUR)"] < 700]
            logger.debug("Filtro applicato per uso studenti.")
        elif usage_value == "uso quotidiano":
            logger.debug("Nessun filtro specifico applicato per uso quotidiano.")
    
    # Filtro aggiuntivo se il messaggio contiene "economico" o "sotto"
    user_message = tracker.latest_message.get("text", "").lower()
    if "economico" in user_message or "sotto" in user_message:
        df_filtered = df_filtered[df_filtered["price(in EUR)"] < 500]
        logger.debug("Filtro aggiuntivo: laptop economico.")
    
    logger.debug(f"Filtraggio avanzato completato. {len(df_filtered)} laptop trovati.")
    return df_filtered

def advanced_sorting(filtered_df: pd.DataFrame, user_message: str) -> pd.DataFrame:
    """
    Ordina il DataFrame filtrato:
    - Se il messaggio contiene "migliore" o "top", ordina per rating decrescente.
    - Altrimenti, ordina per prezzo crescente.
    """
    if "migliore" in user_message or "top" in user_message:
        sorted_df = filtered_df.sort_values(by="rating", ascending=False)
        logger.debug("Ordinamento applicato: rating decrescente.")
    else:
        sorted_df = filtered_df.sort_values(by="price(in EUR)", ascending=True)
        logger.debug("Ordinamento applicato: prezzo crescente.")
    return sorted_df

# -------------------- Azioni Personalizzate --------------------
class ActionSuggerisciLaptop(Action):
    def name(self) -> Text:
        return "action_suggerisci_laptop"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict]:
        user_message = tracker.latest_message.get("text", "").lower()
        logger.debug(f"Messaggio utente per ricerca laptop: {user_message}")
        try:
            filtered_laptops = advanced_filtering(tracker)
            logger.debug(f"Laptop trovati dopo filtraggio: {len(filtered_laptops)}")
            sorted_laptops = advanced_sorting(filtered_laptops, user_message)
            logger.debug(f"Laptop trovati dopo ordinamento: {len(sorted_laptops)}")
            
            limit_slot = tracker.get_slot("limit")
            if limit_slot:
                try:
                    limit_value = int(float(limit_slot))
                    results = sorted_laptops.head(limit_value)
                    logger.debug(f"Limitato a {limit_value} risultati.")
                except Exception as e:
                    logger.error(f"Errore nella conversione del limite: {e}")
                    results = sorted_laptops.head(3)
            else:
                results = sorted_laptops.head(3)
                
            if results.empty:
                risposta = "Mi dispiace, non ho trovato laptop che soddisfino i criteri specificati."
            else:
                risposta = "Ecco alcuni laptop che potrebbero interessarti:\n\n"
                for _, row in results.iterrows():
                    dettagli = format_laptop_details(row)
                    risposta += f"- **{row['name']}**:\n{dettagli}\n\n"
            dispatcher.utter_message(text=risposta)
        except Exception as e:
            logger.exception(f"Errore in ActionSuggerisciLaptop: {e}")
            dispatcher.utter_message(text=f"Si è verificato un errore durante la ricerca dei laptop: {e}")
        return []

class ActionConfrontaLaptop(Action):
    def name(self) -> Text:
        return "action_confronta_laptop"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict]:
        laptop1_input = tracker.get_slot("laptop1")
        laptop2_input = tracker.get_slot("laptop2")
        logger.debug(f"Richiesta confronto tra: {laptop1_input} e {laptop2_input}")
        
        if not laptop1_input or not laptop2_input:
            dispatcher.utter_message(text="Per favore, specifica i nomi di entrambi i laptop da confrontare.")
            return []
        
        matches1 = df[df["name"].str.contains(laptop1_input, case=False, na=False)]
        if matches1.empty:
            dispatcher.utter_message(text=f"Non sono riuscito a trovare un laptop che corrisponda a '{laptop1_input}'.")
            return []
        elif len(matches1) > 1:
            response = f"Ho trovato più laptop che corrispondono a '{laptop1_input}':\n"
            for _, row in matches1.iterrows():
                response += f"- {row['name']}\n"
            response += "\nPer favore, specifica meglio il nome del primo laptop."
            dispatcher.utter_message(text=response)
            return []

        matches2 = df[df["name"].str.contains(laptop2_input, case=False, na=False)]
        if matches2.empty:
            dispatcher.utter_message(text=f"Non sono riuscito a trovare un laptop che corrisponda a '{laptop2_input}'.")
            return []
        elif len(matches2) > 1:
            response = f"Ho trovato più laptop che corrispondono a '{laptop2_input}':\n"
            for _, row in matches2.iterrows():
                response += f"- {row['name']}\n"
            response += "\nPer favore, specifica meglio il nome del secondo laptop."
            dispatcher.utter_message(text=response)
            return []

        data1 = matches1.iloc[0]
        data2 = matches2.iloc[0]
        response = f"Confronto dettagliato tra **{data1['name']}** e **{data2['name']}**:\n\n"
        response += f"### {data1['name']}\n{format_laptop_details(data1)}\n\n"
        response += f"### {data2['name']}\n{format_laptop_details(data2)}\n"
        dispatcher.utter_message(text=response)
        return []

class ActionCaratteristicheLaptop(Action):
    def name(self) -> Text:
        return "action_caratteristiche_laptop"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict]:
        # Applica i filtri comuni basati sugli slot
        laptop_input = tracker.get_slot("laptop")
        logger.debug(f"Richiesta dettagli per laptop: {laptop_input}")
        
        if not laptop_input:
            dispatcher.utter_message(text="Per favore, specifica il nome del laptop di cui vuoi conoscere le caratteristiche.")
            return []

        try:
            matches = apply_common_filters(tracker, df)
            
            if matches.empty:
                logger.error(f"Nessun laptop trovato per: {laptop_input} con i filtri specificati.")
                dispatcher.utter_message(text="Non sono riuscito a trovare il laptop specificato con le caratteristiche richieste. Controlla il nome e gli altri parametri e riprova.")
            elif len(matches) == 1:
                data = matches.iloc[0]
                response = f"Dettagli completi di **{data['name']}**:\n\n{format_laptop_details(data)}"
                dispatcher.utter_message(text=response)
            else:
                response = "Ho trovato più laptop che corrispondono ai criteri inseriti. Per favore, scegli uno tra i seguenti:\n"
                for _, row in matches.iterrows():
                    response += f"- {row['name']}\n"
                dispatcher.utter_message(text=response)
        except Exception as e:
            logger.exception(f"Errore in ActionCaratteristicheLaptop: {e}")
            dispatcher.utter_message(text="Si è verificato un errore durante la ricerca delle caratteristiche del laptop.")
        return []

class ActionPrezzoLaptop(Action):
    def name(self) -> Text:
        return "action_prezzo_laptop"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict]:
        laptop_input = tracker.get_slot("laptop")
        logger.debug(f"Richiesta prezzo per laptop: {laptop_input}")
        if not laptop_input:
            dispatcher.utter_message(text="Per favore, specifica il nome del laptop per conoscere il prezzo.")
            return []
        try:
            matches = df.copy()
            matches = matches[matches["name"].str.contains(laptop_input, case=False, na=False)]
            
            # Applica filtri aggiuntivi se specificati
            for slot in ["ram", "processor", "storage", "os"]:
                value = tracker.get_slot(slot)
                if value:
                    matches = matches[matches[slot].str.contains(value, case=False, na=False)]
            
            usage_slot = tracker.get_slot("usage")
            if usage_slot:
                usage_value = usage_slot.lower()
                if usage_value == "gaming":
                    matches = matches[matches["processor"].str.contains("Core i5|Core i7|Ryzen 7|Ryzen 9", case=False, na=False)]
                elif usage_value == "lavoro":
                    matches = matches[matches["os"].str.contains("Windows", case=False, na=False)]
                elif usage_value == "programmazione":
                    matches = matches[matches["processor"].str.contains("Core i5|Core i7|Ryzen 5|Ryzen 7", case=False, na=False)]
                elif usage_value in ["editing video", "rendering"]:
                    matches = matches[matches["processor"].str.contains("Core i9|Ryzen 9", case=False, na=False)]
                elif usage_value == "studenti":
                    matches = matches[matches["price(in EUR)"] < 700]
            
            if matches.empty:
                response = "Non sono riuscito a trovare il laptop specificato con i filtri forniti. Controlla il nome e gli altri parametri e riprova."
            elif len(matches) == 1:
                prezzo = matches.iloc[0]["price(in EUR)"]
                response = f"Il prezzo di **{matches.iloc[0]['name']}** è {prezzo} EUR."
            else:
                response = "Ho trovato più laptop corrispondenti. Ecco le opzioni:\n"
                for _, row in matches.iterrows():
                    response += f"- {row['name']}: {row['price(in EUR)']} EUR\n"
                response += "\nPer favore, specifica meglio il nome o i parametri del laptop."
            dispatcher.utter_message(text=response)
        except Exception as e:
            logger.error(f"Errore in ActionPrezzoLaptop: {e}")
            dispatcher.utter_message(text="Si è verificato un errore durante la ricerca del prezzo del laptop.")
        return []

class ActionHelp(Action):
    def name(self) -> Text:
        return "action_help"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict]:
        message = (
            "Sono qui per aiutarti a trovare il laptop ideale! Puoi chiedermi di confrontare due laptop, "
            "cercare in base a specifiche tecniche, visualizzare immagini, conoscere il prezzo e molto altro.\n"
            "Ad esempio, prova a dire:\n"
            " - 'Confronta il Dell Inspiron con l'HP Pavilion'\n"
            " - 'Dimmi le specifiche del MacBook Air'\n"
            " - 'Quanto costa l'ASUS ZenBook?'\n"
            " - 'Mostrami l'immagine del Lenovo IdeaPad'\n"
            " - 'Aiuto'\n"
        )
        dispatcher.utter_message(text=message)
        return []

class ActionShowLaptopList(Action):
    def name(self) -> Text:
        return "action_show_laptop_list"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict]:
        logger.debug("Richiesta lista laptop.")
        try:
            laptops = df.head(5)
            response = "Ecco una lista di alcuni laptop disponibili:\n\n"
            for _, row in laptops.iterrows():
                response += (
                    f"- **{row['name']}**: {row['processor']}, {row['ram']}, {row['os']}, "
                    f"{row['storage']}, Display: {row['display(in inch)']} pollici, Prezzo: {row['price(in EUR)']} EUR\n\n"
                )
            dispatcher.utter_message(text=response)
        except Exception as e:
            logger.error(f"Errore nel recupero della lista laptop: {e}")
            dispatcher.utter_message(text="Si è verificato un errore nel recupero della lista dei laptop.")
        return []

class ActionMostraImmagineLaptop(Action):
    def name(self) -> Text:
        return "action_mostra_immagine_laptop"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict]:
        laptop_input = tracker.get_slot("laptop")
        logger.debug(f"Richiesta di mostrare immagine per: {laptop_input}")

        if not laptop_input:
            dispatcher.utter_message(text="Per favore, specifica il nome del laptop di cui vuoi vedere l'immagine.")
            return []

        matches = df[df["name"].str.contains(laptop_input, case=False, na=False)]
        if matches.empty:
            dispatcher.utter_message(text="Non sono riuscito a trovare il laptop specificato. Controlla il nome e riprova.")
        elif len(matches) == 1:
            img_link = matches.iloc[0].get("img_link", None)
            if img_link and isinstance(img_link, str) and img_link.strip():
                dispatcher.utter_message(text="Ecco l'immagine del laptop:", image=img_link)
            else:
                dispatcher.utter_message(text="Mi dispiace, non ho trovato un link all'immagine per questo laptop.")
        else:
            response = "Ho trovato più laptop corrispondenti. Ecco le immagini disponibili:\n"
            images_found = False
            for _, row in matches.iterrows():
                img_link = row.get("img_link", None)
                if img_link and isinstance(img_link, str) and img_link.strip():
                    response += f"- {row['name']}: {img_link}\n"
                    images_found = True
            if images_found:
                dispatcher.utter_message(text=response)
            else:
                dispatcher.utter_message(text="Ho trovato più laptop, ma nessuno presenta un link valido all'immagine.")
        return []
    
class ActionResetSlots(Action):
    def name(self) -> Text:
        return "action_reset_slots"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict]:
        events = [SlotSet(slot, None) for slot in tracker.current_slot_values().keys()]
        dispatcher.utter_message(text="Tutti gli slot sono stati resettati.")
        return events