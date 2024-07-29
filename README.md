# Fine-tuned LoRA Selector met TinyLlama en Embeddings

Dit project implementeert een systeem dat LoRA adapters selecteert en gebruikt op basis van fine-tuned embeddings en een TinyLlama taalmodel. Het systeem is specifiek ontworpen om te differentiëren tussen verschillende onderwerpen, zoals steden en geschiedenis.

## Overzicht

Het systeem werkt als volgt:
1. Fine-tuned embeddings model categoriseert input in onderwerpen (als voorbeeld steden of geschiedenis).
2. Op basis van de hoogste embeddings waarde wordt de beste LoRA adapter geselecteerd.
3. De geselecteerde LoRA wordt toegepast op het TinyLlama model voor gespecialiseerde inferentie.

## Installatie

1. Clone de repository en navigeer naar de projectmap:

```bash
git clone https://github.com/s-smits/lora-selector
cd lora-selector
```

2. Maak een virtuele omgeving aan met de naam `venv_lora_selector` en activeer deze:

Voor macOS en Linux:
```bash
python3 -m venv venv_lora_selector
source venv_lora_selector/bin/activate
```

Voor Windows:
```bash
python -m venv venv_lora_selector
venv_lora_selector\Scripts\activate
```

3. Installeer de vereiste dependencies:

```bash
pip install -r requirements.txt
```

## Gebruik

1. Pas het `training_data.json` bestand aan (momenteel voorbeelden van steden- en geschiedenisvragen):

```json
{
  "text": [
    {"question": "Wat is de hoofdstad van Nederland?", "answer": "Amsterdam", "subject": "cities"},
    {"question": "In welk jaar vond de Franse Revolutie plaats?", "answer": "1789", "subject": "history"}
  ]
}
```

2. Start het main script:

```bash
python main.py
```

Dit script zal:
- Het embeddings model fine-tunen om onderscheid te maken tussen steden- en geschiedenisvragen.
- LoRA adapters trainen voor zowel steden als geschiedenis.
- Inference met de beste LoRA adapter te kiezen op basis van de hoogste embeddings waarde.

## Voorbeeld

Input: "Welke stad staat bekend om zijn scheve toren?"
1. Embeddings model categoriseert dit als een stedenvraag.
2. Steden-LoRA wordt geselecteerd vanwege de hoogste embeddings waarde.
3. TinyLlama met steden-LoRA genereert een antwoord: "De stad die bekend staat om zijn scheve toren is Pisa, Italië."

Input: "Wie was de eerste president van de Verenigde Staten?"
1. Embeddings model categoriseert dit als een geschiedenisvraag.
2. Geschiedenis-LoRA wordt geselecteerd vanwege de hoogste embeddings waarde.
3. TinyLlama met geschiedenis-LoRA genereert een antwoord: "De eerste president van de Verenigde Staten was George Washington."

## Structuur

Het project bestaat uit twee hoofdbestanden:

1. `main.py`: Bevat de logica voor het trainen van LoRA adapters en inference.
2. `finetune_embeddings_model.py`: Bevat de code voor het fine-tunen van het embeddings model om onderscheid te maken tussen onderwerpen.

## Tailor-made

- Voeg zelf onderwerpen toe via `training_data.json`.
- Pas de `base_language_model_name` en `base_embeddings_model_name` variabelen aan in `main.py` voor andere modellen.
- Optimaliseer de LoRA training door de hyperparameters in de `LoraConfig` in `main.py` aan te passen.

## Opmerkingen

- Dit project maakt met name gebruik van PyTorch en Transformers.
- Zorg voor voldoende GPU/CPU-geheugen, vooral bij het toevoegen van meer onderwerpen (16GB).
- Back-end `device` wordt automatisch gekozen (Cuda > MPS > CPU).