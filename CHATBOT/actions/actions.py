import pandas as pd
import logging
from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet
from rasa_sdk import FormValidationAction
from rasa_sdk.types import DomainDict
import re

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Carica il dataset
try:
    df = pd.read_csv("laptops_etl_clean.csv")
    logger.info("Dataset laptop caricato correttamente.")
except Exception as e:
    logger.error(f"Errore nel caricamento del dataset: {e}")
    df = pd.DataFrame()

# Formatta i dettagli dei laptop
def format_laptop_details(row: pd.Series) -> str:
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

def get_processor_score(proc: str) -> int:
    proc = proc.lower()
    if "apple" in proc:
        if "m1 max" in proc or "m2 max" in proc:
            return 4
        elif "m1 pro" in proc or "m2 pro" in proc:
            return 3
        elif "m1" in proc or "m2" in proc:
            return 2
    elif "intel" in proc:
        if "i9" in proc:
            return 4
        elif "i7" in proc:
            return 3
        elif "i5" in proc:
            return 2
        elif "i3" in proc:
            return 1
    elif "amd" in proc:
        if "ryzen 9" in proc:
            return 4
        elif "ryzen 7" in proc:
            return 3
        elif "ryzen 5" in proc:
            return 2
        elif "ryzen 3" in proc:
            return 1
    return 0

def extract_ram_value(ram_str: str) -> int:
    # Estrae il primo numero presente (es. "8" da "8 GB DDR4 RAM")
    match = re.search(r'(\d+)', ram_str)
    return int(match.group(1)) if match else 0

def compare_laptop_performance(row1: pd.Series, row2: pd.Series) -> str:
    # Ottiene punteggi per processore
    proc_score1 = get_processor_score(str(row1.get("processor", "")))
    proc_score2 = get_processor_score(str(row2.get("processor", "")))
    
    # Estrae e converte la RAM in GB
    ram_value1 = extract_ram_value(str(row1.get("ram", "")))
    ram_value2 = extract_ram_value(str(row2.get("ram", "")))
    
    # Calcola un punteggio cumulativo (ponderando il processore maggiormente)
    score1 = proc_score1 * 10 + ram_value1
    score2 = proc_score2 * 10 + ram_value2
    
    if score1 > score2:
        return f"In base ai calcoli, **{row1['name']}** sembra essere più performante di **{row2['name']}**."
    elif score2 > score1:
        return f"In base ai calcoli, **{row2['name']}** sembra essere più performante di **{row1['name']}**."
    else:
        return "I due laptop sembrano offrire prestazioni simili."

# Applica filtri comuni a tutte le azioni
def apply_common_filters(tracker: Tracker, df_input: pd.DataFrame) -> pd.DataFrame:
    df_filtered = df_input.copy()
    
    # Filtra per il nome (se fornito)
    laptop_input = tracker.get_slot("laptop")
    if laptop_input:
        df_filtered = df_filtered[df_filtered["name"].str.contains(laptop_input, case=False, na=False, regex=False)]
    
    # Filtra per altri slot espliciti (ram, processor, storage, os)
    for slot in ["ram1", "processor1", "storage1"]:
        value = tracker.get_slot(slot)
        if value:
            df_filtered = df_filtered[df_filtered[slot].str.contains(value, case=False, na=False, regex=False)]
    
    # Se lo slot "usage" è valorizzato, applichiamo dei filtri predefiniti basati su di esso
    usage_slot = tracker.get_slot("usage")
    if usage_slot:
        usage = usage_slot.lower()
        if usage == "gaming":
            # Esempio: per il gaming vogliamo processori più potenti e almeno 16 GB di RAM
            df_filtered = df_filtered[df_filtered["processor"].str.contains("Core i5|Core i7|Ryzen 7|Ryzen 9", case=False, na=False)]
            df_filtered = df_filtered[df_filtered["ram"].str.contains("16", case=False, na=False)]
        elif usage == "lavoro":
            # Per il lavoro si preferisce il sistema operativo Windows
            df_filtered = df_filtered[df_filtered["os"].str.contains("Windows", case=False, na=False)]
            df_filtered = df_filtered[df_filtered["processor"].str.contains("Core i5|Ryzen 7", case=False, na=False)]
        elif usage == "programmazione":
            # Per la programmazione processori equilibrati
            df_filtered = df_filtered[df_filtered["processor"].str.contains("Core i7|Core i9|Ryzen 7|Ryzen 9", case=False, na=False)]
        elif usage in ["editing video", "rendering"]:
            # Per l'editing video vogliamo i top: processori molto potenti e almeno 16 GB di RAM
            df_filtered = df_filtered[df_filtered["processor"].str.contains("Core i9|Ryzen 9", case=False, na=False)]
            df_filtered = df_filtered[df_filtered["ram"].str.contains("16|32", case=False, na=False)]
        elif usage == "studenti":
            # Per gli studenti si può imporre un budget ridotto: prezzo inferiore a 700 EUR
            try:
                df_filtered["price(in EUR)"] = pd.to_numeric(df_filtered["price(in EUR)"], errors="coerce")
                df_filtered = df_filtered[df_filtered["price(in EUR)"] < 700]
                df_filtered = df_filtered[df_filtered["processor"].str.contains("Core i7|Ryzen 7", case=False, na=False)]
            except Exception as e:
                logger.error(f"Errore conversione prezzo: {e}")
        elif usage == "uso quotidiano":
            # Per uso quotidiano, nessun filtro aggiuntivo
            pass
    
    # Filtra per prezzo (se fornito)
    price_condition = tracker.get_slot("price_condition")
    price_range = tracker.get_slot("price_range")
    if price_condition and price_range:
            if price_condition == "inferiore":
                df_filtered = df_filtered[df_filtered["price(in EUR)"] < price_range]
            elif price_condition == "superiore":
                df_filtered = df_filtered[df_filtered["price(in EUR)"] > price_range]
            elif price_condition == "uguale":
                df_filtered = df_filtered[df_filtered["price(in EUR)"] == price_range]
            elif price_condition == "sotto":
                df_filtered = df_filtered[df_filtered["price(in EUR)"] < price_range]
            elif price_condition == "sopra":
                df_filtered = df_filtered[df_filtered["price(in EUR)"] > price_range]
            elif price_condition == "meno":
                df_filtered = df_filtered[df_filtered["price(in EUR)"] < price_range]
            elif price_condition == "più":
                df_filtered = df_filtered[df_filtered["price(in EUR)"] > price_range]
            elif price_condition == "minore":
                df_filtered = df_filtered[df_filtered["price(in EUR)"] < price_range]
            elif price_condition == "maggiore":
                df_filtered = df_filtered[df_filtered["price(in EUR)"] > price_range]
            else:
                logger.warning(f"Condizione prezzo non riconosciuta: {price_condition}")
    elif price_range and not price_condition:
        df_filtered = df_filtered[df_filtered["price(in EUR)"] <= price_range]

    rating = tracker.get_slot("rating_range")
    if rating:
        try:
            rating = float(rating)
            df_filtered = df_filtered[df_filtered["rating"] >= rating]
        except ValueError:
            logger.warning(f"Valore di rating non valido: {rating}")

    os = tracker.get_slot("os")
    if os:
        df_filtered = df_filtered[df_filtered["os"].str.contains(os, case=False, na=False, regex=False)]

    brand_value = tracker.get_slot("brand")
    if brand_value:
        df_filtered = df_filtered[df_filtered["name"].str.contains(brand_value, case=False, na=False, regex=False)]

    return df_filtered

# Reset slot
class ActionResetSlots(Action):
    def name(self) -> Text:
        return "action_reset_slots"
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict]:
        slots_to_reset = [
            "laptop", "usage", "price_range", "price_condition",
            "laptop1", "processor1", "ram1", "storage1",
            "laptop2", "processor2", "ram2", "storage2"
        ]
        return [SlotSet(slot, None) for slot in slots_to_reset]
    
# Validazione della form di ricerca per ricerca dei laptop
class ValidateLaptopSearchForm(FormValidationAction):
    def name(self) -> Text:
        return "validate_laptop_search_form"

    def validate_price_range(
        self,
        slot_value: Any,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict
    ) -> Dict[Text, Any]:
        try:
            val = float(slot_value)
        except ValueError:
            dispatcher.utter_message(text="Inserisci un valore numerico valido per il prezzo.")
            return {"price_range": None}
        if not val:
            dispatcher.utter_message(text="Per favore, specifica un prezzo massimo.")
            return {"price_range": None}
        elif val <= 0:
            dispatcher.utter_message(text="Il prezzo deve essere maggiore di zero.")
            return {"price_range": None}

        return {"price_range": val}

# Validazione della form per il confronto di due laptop
class ValidateLaptopCompareForm(FormValidationAction):
    def name(self) -> Text:
        return "validate_laptop_compare_form"

    def validate_laptop1(
        self,
        slot_value: Any,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict
    ) -> Dict[Text, Any]:
        # Controlla se il form sta chiedendo il nome del primo laptop
        if tracker.get_slot("requested_slot") == "laptop1":
            if not slot_value or slot_value.strip() == "":
                dispatcher.utter_message(text="Per favore, indica il nome del primo laptop.")
                return {"laptop1": None}
            return {"laptop1": slot_value}
        # Se il form non richiede laptop1 in questo momento, mantieni il valore corrente
        return {"laptop1": tracker.get_slot("laptop1")}

    def validate_laptop2(
        self,
        slot_value: Any,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict
    ) -> Dict[Text, Any]:
        # Controlla se il form sta chiedendo il nome del secondo laptop
        if tracker.get_slot("requested_slot") == "laptop2":
            if not slot_value or slot_value.strip() == "":
                dispatcher.utter_message(text="Per favore, indica il nome del secondo laptop.")
                return {"laptop2": None}
            return {"laptop2": slot_value}
        return {"laptop2": tracker.get_slot("laptop2")}

    def validate_processor(
        self,
        slot_value: Any,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict
    ) -> Dict[Text, Any]:
        requested = tracker.get_slot("requested_slot")
        if requested == "processor1":
            if not slot_value or slot_value.strip() == "":
                dispatcher.utter_message(text="Per favore, indica il processore del primo laptop.")
                return {"processor1": None}
            return {"processor1": slot_value}
        elif requested == "processor2":
            if not slot_value or slot_value.strip() == "":
                dispatcher.utter_message(text="Per favore, indica il processore del secondo laptop.")
                return {"processor2": None}
            return {"processor2": slot_value}
        else:
            if not tracker.get_slot("processor1"):
                return {"processor1": slot_value}
            elif not tracker.get_slot("processor2"):
                return {"processor2": slot_value}
        return {}

    def validate_ram(
        self,
        slot_value: Any,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict
    ) -> Dict[Text, Any]:
        requested = tracker.get_slot("requested_slot")
        if requested == "ram1":
            if not slot_value or slot_value.strip() == "":
                dispatcher.utter_message(text="Per favore, indica la quantità di RAM del primo laptop.")
                return {"ram1": None}
            return {"ram1": slot_value}
        elif requested == "ram2":
            if not slot_value or slot_value.strip() == "":
                dispatcher.utter_message(text="Per favore, indica la quantità di RAM del secondo laptop.")
                return {"ram2": None}
            return {"ram2": slot_value}
        else:
            if not tracker.get_slot("ram1"):
                return {"ram1": slot_value}
            elif not tracker.get_slot("ram2"):
                return {"ram2": slot_value}
        return {}

    def validate_storage(
        self,
        slot_value: Any,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict
    ) -> Dict[Text, Any]:
        requested = tracker.get_slot("requested_slot")
        if requested == "storage1":
            if not slot_value or slot_value.strip() == "":
                dispatcher.utter_message(text="Per favore, indica il tipo di storage del primo laptop.")
                return {"storage1": None}
            return {"storage1": slot_value}
        elif requested == "storage2":
            if not slot_value or slot_value.strip() == "":
                dispatcher.utter_message(text="Per favore, indica il tipo di storage del secondo laptop.")
                return {"storage2": None}
            return {"storage2": slot_value}
        else:
            if not tracker.get_slot("storage1"):
                return {"storage1": slot_value}
            elif not tracker.get_slot("storage2"):
                return {"storage2": slot_value}
        return {}

# Validazione della form per il laptop migliore da consigliare
class ValidateLaptopRecommendForm(FormValidationAction):
    def name(self) -> Text:
        return "validate_laptop_recommend_form"

    def validate_usage(
        self,
        slot_value: Any,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict
    ) -> Dict[Text, Any]:
        if not slot_value:
            dispatcher.utter_message(text="Per favore, specifica l'uso (es. 'gaming', 'lavoro', 'studenti', ecc.).")
            return {"usage": None}
        return {"usage": slot_value}

    def validate_price_range(
        self,
        slot_value: Any,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict
    ) -> Dict[Text, Any]:
        try:
            val = float(slot_value)
        except ValueError:
            dispatcher.utter_message(text="Inserisci un valore numerico valido per il prezzo.")
            return {"price_range": None}

        if val <= 0:
            dispatcher.utter_message(text="Il prezzo deve essere maggiore di zero.")
            return {"price_range": None}
        return {"price_range": val}

class ActionSuggerisciLaptop(Action):
    def name(self) -> Text:
        return "action_suggerisci_laptop"
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict]:
        user_message = tracker.latest_message.get("text", "").lower()
        logger.debug(f"Ricerca laptop con criteri: {user_message}")
        try:
            filtered_laptops = apply_common_filters(tracker, df)
            sorted_laptops = filtered_laptops.sort_values(by="price(in EUR)", ascending=True)
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
        return [SlotSet(slot, None) for slot in tracker.current_slot_values().keys()]

class ActionConfrontaLaptop(Action):
    def name(self) -> Text:
        return "action_confronta_laptop"
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict]:
        laptop1_input = tracker.get_slot("laptop1")
        processor1 = tracker.get_slot("processor1")
        ram1 = tracker.get_slot("ram1")
        storage1 = tracker.get_slot("storage1")
        laptop2_input = tracker.get_slot("laptop2")
        processor2 = tracker.get_slot("processor2")
        ram2 = tracker.get_slot("ram2")
        storage2 = tracker.get_slot("storage2")

        logger.debug(f"Confronto tra: {laptop1_input} (proc: {processor1}, ram: {ram1}, storage: {storage1}) e {laptop2_input} (proc: {processor2}, ram: {ram2}, storage: {storage2})")

        if not laptop1_input or not laptop2_input:
            dispatcher.utter_message(text="Per favore, specifica i nomi di entrambi i laptop da confrontare, includendo le specifiche per disambiguare se necessario.")
            return []
        
        matches1 = df[df["name"].str.contains(laptop1_input, case=False, na=False, regex=False)]
        if processor1:
            matches1 = matches1[matches1["processor"].str.contains(processor1, case=False, na=False, regex=False)]
        if ram1:
            matches1 = matches1[matches1["ram"].str.contains(ram1, case=False, na=False, regex=False)]
        if storage1:
            matches1 = matches1[
                matches1["storage"].str.contains(storage1, case=False, na=False, regex=False) &
                (matches1["storage"].str.len() <= (len(storage1) + 4))
            ]
        if matches1.empty:
            dispatcher.utter_message(text=f"Non ho trovato alcun laptop per '{laptop1_input}' con le specifiche indicate. \nPer riprendere il confronto, specifica nuovamente i nomi dei due laptop.")
            return [
                SlotSet("laptop1", None),
                SlotSet("processor1", None),
                SlotSet("ram1", None),
                SlotSet("storage1", None)
            ]
        elif len(matches1) > 1:
            response = f"Ho trovato più laptop per '{laptop1_input}' con le specifiche fornite:\n"
            for _, row in matches1.iterrows():
                response += f"- {row['name']} (Processor: {row.get('processor', 'N/A')}, RAM: {row.get('ram', 'N/A')}, Storage: {row.get('storage', 'N/A')})\n"
            response += "\nPer favore, specifica meglio il primo laptop.\nPer riprendere il confronto, specifica nuovamente i nomi dei due laptop."
            dispatcher.utter_message(text=response)
            return [
                SlotSet("laptop1", None),
                SlotSet("processor1", None),
                SlotSet("ram1", None),
                SlotSet("storage1", None)
            ]
        
        matches2 = df[df["name"].str.contains(laptop2_input, case=False, na=False, regex=False)]
        if processor2:
            matches2 = matches2[matches2["processor"].str.contains(processor2, case=False, na=False, regex=False)]
        if ram2:
            matches2 = matches2[matches2["ram"].str.contains(ram2, case=False, na=False, regex=False)]
        if storage2:
            matches2 = matches2[
                matches2["storage"].str.contains(storage2, case=False, na=False, regex=False) &
                (matches2["storage"].str.len() <= (len(storage2) + 4))
            ]
        if matches2.empty:
            dispatcher.utter_message(text=f"Non ho trovato alcun laptop per '{laptop2_input}' con le specifiche indicate. \nPer riprendere il confronto, specifica nuovamente i nomi dei due laptop.")
            return [
                SlotSet("laptop2", None),
                SlotSet("processor2", None),
                SlotSet("ram2", None),
                SlotSet("storage2", None)
            ]
        elif len(matches2) > 1:
            response = f"Ho trovato più laptop per '{laptop2_input}' con le specifiche fornite:\n"
            for _, row in matches2.iterrows():
                response += f"- {row['name']} (Processor: {row.get('processor', 'N/A')}, RAM: {row.get('ram', 'N/A')}, Storage: {row.get('storage', 'N/A')})\n"
            response += "\nPer favore, specifica meglio il secondo laptop. \nPer riprendere il confronto, specifica nuovamente i nomi dei due laptop."
            dispatcher.utter_message(text=response)
            return [
                SlotSet("laptop2", None),
                SlotSet("processor2", None),
                SlotSet("ram2", None),
                SlotSet("storage2", None)
            ]
        data1 = matches1.iloc[0]
        data2 = matches2.iloc[0]
        response = (
            f"Confronto dettagliato tra **{data1['name']}** e **{data2['name']}**:\n\n"
            f"### {data1['name']}\n{format_laptop_details(data1)}\n\n"
            f"### {data2['name']}\n{format_laptop_details(data2)}\n\n"
        )
        performance_analysis = compare_laptop_performance(data1, data2)
        response += performance_analysis
        
        dispatcher.utter_message(text=response)
        return [
            SlotSet(slot, None)
            for slot in tracker.current_slot_values().keys()
        ]

# Consiglia un laptop con rating 5.0\
class ActionConsigliaLaptop(Action):
    def name(self) -> Text:
        return "action_consiglia_laptop"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict]:
        user_message = tracker.latest_message.get("text", "").lower()
        logger.debug(f"Richiesta consiglio laptop con criteri: {user_message}")
        try:
            # Applica i filtri comuni (uso, prezzo, ecc.)
            filtered_laptops = apply_common_filters(tracker, df)
            
            # Filtra per rating fisso: solo laptop con rating tra 4.5 e 5.0
            # Assicurati che la colonna "rating" venga convertita in float
            filtered_laptops["rating"] = pd.to_numeric(filtered_laptops["rating"], errors="coerce")
            filtered_laptops = filtered_laptops[
                (filtered_laptops["rating"] >= 4.5) & (filtered_laptops["rating"] <= 5.0)
            ]
            
            # Ordina per prezzo crescente e prendi il primo risultato
            sorted_laptops = filtered_laptops.sort_values(by="price(in EUR)", ascending=True)
            results = sorted_laptops.head(1)
            
            if results.empty:
                risposta = "Mi dispiace, non ho trovato laptop che soddisfino i criteri specificati."
            else:
                row = results.iloc[0]
                risposta = f"Ti consiglio di dare un'occhiata a **{row['name']}**:\n{format_laptop_details(row)}"
            
            dispatcher.utter_message(text=risposta)
        except Exception as e:
            logger.exception(f"Errore in ActionConsigliaLaptop: {e}")
            dispatcher.utter_message(text=f"Si è verificato un errore durante il consiglio del laptop: {e}")
        
        # Resetta tutti gli slot per ripulire il contesto
        return []

class ActionCaratteristicheLaptop(Action):
    def name(self) -> Text:
        return "action_caratteristiche_laptop"
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict]:
        laptop_input = tracker.get_slot("laptop")
        if not laptop_input:
            dispatcher.utter_message(text="Per favore, specifica il nome del laptop di cui vuoi conoscere i dettagli.")
            return []
        matches = df[df["name"].str.contains(laptop_input, case=False, na=False, regex=False)]
        
        # Mapping degli slot ai nomi delle colonne nel dataset
        mapping = {
            "processor1": "processor",
            "ram1": "ram",
            "storage1": "storage"
        }
        
        for slot, column in mapping.items():
            value = tracker.get_slot(slot)
            if value:
                matches = matches[matches[column].str.contains(value, case=False, na=False, regex=False)]
                
        if matches.empty:
            dispatcher.utter_message(text=f"Non ho trovato dettagli per il laptop '{laptop_input}'.")
            return []
        elif len(matches) > 1:
            response = f"Ho trovato più laptop che corrispondono a '{laptop_input}':\n"
            for _, row in matches.iterrows():
                response += f"- {row['name']} (Processor: {row.get('processor', 'N/A')}, RAM: {row.get('ram', 'N/A')}, Storage: {row.get('storage', 'N/A')})\n"
            response += "\nPer favore, specifica meglio il nome o aggiungi filtri."
            dispatcher.utter_message(text=response)
            return []
        data = matches.iloc[0]
        response = f"Dettagli completi di **{data['name']}**:\n\n{format_laptop_details(data)}"
        dispatcher.utter_message(text=response)
        return [SlotSet(slot, None) for slot in tracker.current_slot_values().keys()]

class ActionMostraImmagineLaptop(Action):
    def name(self) -> Text:
        return "action_mostra_immagine_laptop"
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict]:
        laptop_input = tracker.get_slot("laptop")
        if not laptop_input:
            dispatcher.utter_message(text="Per favore, specifica il nome del laptop di cui vuoi vedere l'immagine.")
            return []
        matches = df[df["name"].str.contains(laptop_input, case=False, na=False, regex=False)]
        if matches.empty:
            dispatcher.utter_message(text="Non ho trovato il laptop specificato. Controlla il nome e riprova.")
        elif len(matches) == 1:
            img_link = matches.iloc[0].get("img_link", None)
            if img_link and isinstance(img_link, str) and img_link.strip():
                dispatcher.utter_message(text="Ecco l'immagine del laptop:", image=img_link)
            else:
                dispatcher.utter_message(text="Non ho trovato un link valido per l'immagine di questo laptop.")
        else:
            response = "Ho trovato più laptop. Ecco le immagini disponibili:\n"
            images_found = False
            for _, row in matches.iterrows():
                img_link = row.get("img_link", None)
                if img_link and isinstance(img_link, str) and img_link.strip():
                    response += f"- {row['name']}: {img_link}\n"
                    images_found = True
            if images_found:
                dispatcher.utter_message(text=response)
            else:
                dispatcher.utter_message(text="Ho trovato più laptop, ma nessuno ha un link immagine valido.")
        return []

class ActionHelp(Action):
    def name(self) -> Text:
        return "action_help"
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict]:
        message = (
            "👋 Benvenuto nel chatbot dei portatili!\n"
            "💻 Sono qui per aiutarti a trovare il laptop ideale!\n"
            "✨ Puoi:\n"
            "   • Cercare un laptop in base a criteri come uso e prezzo.\n"
            "   • Confrontare due laptop specificando per ciascuno nome e specifiche.\n"
            "   • Ricevere un consiglio su quale laptop ti conviene acquistare.\n"
            "   • Visualizzare l'immagine di un laptop.\n"
            "   • O chiedere i dettagli completi di un laptop.\n"
            "\n"
            "📚 Esempi:\n"
            "   • 'Cerco un laptop per lavoro sotto 800'\n"
            "   • 'Confronta il [Dell Inspiron](laptop1) con il [HP Pavilion](laptop2) con [Intel Core i5](processor1) e [8GB DDR4](ram1) vs [Intel Core i7](processor2) e [16GB DDR4](ram2)'\n"
            "   • 'Mi consigli un laptop per gaming con un budget di 1200'\n"
            "   • 'Dammi i dettagli del [Dell XPS 15](laptop)'\n"
            "   • 'Mostrami l'immagine del [HP Pavilion](laptop)'\n"
            "\n"
            "📝 Dati disponibili per i laptop:\n"
            "   - Esempi di utilizzi per il laptop: 'gaming', 'lavoro', 'programmazione', 'editing video', 'studenti' ed 'uso quotidiano'\n\n"
            "   - Esempi di processori: 'Intel Core i5 Processor (11th Gen)', 'Intel Core i3 Processor (11th Gen)', "
                "'Intel Core i5 Processor (10th Gen)', 'Intel Core i3 Processor (10th Gen)', 'AMD Athlon Dual Core Processor', "
                "'Apple M1 Processor', 'Intel Celeron Dual Core Processor', 'AMD Ryzen 3 Dual Core Processor', "
                "'Intel Core i5 Processor (12th Gen)', 'Intel Core i7 Processor (11th Gen)', 'AMD Ryzen 5 Hexa Core Processor', "
                "'Intel Core i3 Processor (12th Gen)', 'AMD Ryzen 3 Quad Core Processor', 'AMD Ryzen 7 Octa Core Processor', "
                "'Qualcomm Snapdragon 7c Gen 2 Processor', 'Intel Core i7 Processor (12th Gen)', 'Intel Pentium Silver Processor', "
                "'AMD Ryzen 5 Quad Core Processor', 'Intel Core i9 Processor (12th Gen)', 'AMD Dual Core Processor', "
                "'Apple M2 Processor', 'AMD Ryzen 9 Octa Core Processor', 'Apple M1 Max Processor', 'Apple M1 Pro Processor', "
                "'Intel Pentium Quad Core Processor', 'Intel Core i7 Processor (10th Gen)', 'AMD Ryzen 9 Octa Core Processor (5th Gen)', "
                "'Intel Core i9 Processor (10th Gen)', 'Intel Celeron Quad Core Processor', 'AMD Ryzen 5 Dual Core Processor (5th Gen)', "
                "'Intel Core i9 Processor (11th Gen)', 'AMD Ryzen 7 Quad Core Processor', 'Intel Core i5 Processor (7th Gen)', "
                "'Intel Core i5 Processor (9th Gen)', 'AMD Ryzen 7 Hexa Core Processor', 'Intel Core i5 Processor (8th Gen)', "
                "'Intel Core i7 Processor (8th Gen)', 'Intel Core i7 Processor (7th Gen)', 'AMD Ryzen 9 Octa Core Processor (10th Gen)', "
                "'Intel Core i3 Processor (7th Gen)', 'Intel Core i9 Processor (8th Gen)', 'AMD APU Dual Core A6 Processor', "
                "'Intel Celeron Dual Core Processor (4th Gen)', 'Intel Core i5 Processor (5th Gen)', 'Intel Core i5 Processor (4th Gen)', "
                "'Intel Core i7 Processor (9th Gen)', 'MediaTek MediaTek Kompanio 500 Processor'\n\n"
            "   - Esempi di RAM: '16 GB DDR4 RAM', '8 GB DDR4 RAM', '4 GB DDR4 RAM', '16 GB LPDDR5 RAM', "
                "'16 GB DDR5 RAM', '4 GB LPDDR4X RAM', '8 GB LPDDR4X RAM', '32 GB LPDDR5 RAM', '4 GB LPDDR4 RAM', "
                "'16 GB LPDDR4X RAM', '8 GB DDR5 RAM', '8 GB Unified Memory RAM', '32 GB Unified Memory RAM', "
                "'16 GB Unified Memory RAM', '32 GB DDR5 RAM', '32 GB DDR4 RAM', '8 GB DDR3 RAM', '8 GB LPDDR3 RAM', "
                "'16 GB LPDDR3 RAM', '16 GB DDR3 RAM'\n\n"
            "   - Esempi di storage: '512 GB SSD', '1 TB HDD|256 GB SSD', '256 GB SSD', '1 TB SSD', '2 TB SSD', "
                "'1 TB HDD|512 GB SSD', '1 TB HDD', '128 GB SSD', '256 GB HDD|256 GB SSD', '1 TB HDD|128 GB SSD', "
                "'PCI-e SSD (NVMe) ready,Silver-Lining Print Keyboard,Matrix Display (Extend),Cooler Boost 5,Hi-Res Audio,Nahimic 3,144Hz Panel,Thin Bezel,RGB Gaming Keyboard,Speaker Tuning Engine,MSI Center', "
                "'PCI-e Gen4 SSD?SHIFT?Matrix Display (Extend)?Cooler Boost 3?Thunderbolt 4?Finger Print Security?True Color 2.0?Hi-Res Audio?Nahimic 3? 4-Sided Thin bezel?MSI Center?Silky Smooth Touchpad?Military-Grade Durability', "
                "'2 TB HDD', '512 GB HDD|512 GB SSD', '256 GB HDD'\n\n"
        )
        dispatcher.utter_message(text=message)
        return []

class ActionShowLaptopList(Action):
    def name(self) -> Text:
        return "action_show_laptop_list"
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict]:
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
            logger.error(f"Errore nel recupero della lista: {e}")
            dispatcher.utter_message(text="Si è verificato un errore nel recupero della lista dei laptop.")
        return [SlotSet(slot, None) for slot in tracker.current_slot_values().keys()]
