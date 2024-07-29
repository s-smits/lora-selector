Zeker, hier is een aangepaste handleiding voor de code die je hebt gedeeld:

# Fine-tuned LoRA Selector met TinyLlama en BGE Embeddings

Dit project implementeert een systeem dat LoRA adapters selecteert en gebruikt op basis van fine-tuned embeddings en een TinyLlama taalmodel.

## Installatie

1. Clone de repository en navigeer naar de projectmap:

```bash
git clone https://github.com/jouw-username/lora-selector
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

1. Zorg ervoor dat je een `training_data.json` bestand hebt in de hoofdmap van het project met de juiste trainingsgegevens.

2. Start het hoofdscript:

```bash
python main.py
```

Dit script zal:
- Het embeddings model fine-tunen
- LoRA adapters trainen
- Inferentie uitvoeren met het beste LoRA adapter voor elke invoer

## Structuur

Het project bestaat uit twee hoofdbestanden:

1. `main.py`: Bevat de hoofdlogica voor het trainen van LoRA adapters en het uitvoeren van inferentie.
2. `finetune_embeddings_model.py`: Bevat de code voor het fine-tunen van het embeddings model.

## Aanpassen

- Pas de `base_language_model_name` en `base_embeddings_model_name` variabelen aan in `main.py` als je andere modellen wilt gebruiken.
- Pas de hyperparameters aan in de `LoraConfig` in `main.py` om de LoRA training te optimaliseren.
- Pas de `finetune_embeddings_model` functie in `finetune_embeddings_model.py` aan om de embeddings training te wijzigen.

## Opmerkingen

- Dit project maakt gebruik van PyTorch en de Transformers bibliotheek.
- Zorg ervoor dat je voldoende GPU-geheugen hebt voor het trainen van de modellen.
- De code is ingesteld om MPS (Metal Performance Shaders) te gebruiken op compatibele Apple-apparaten. Pas de `device` variabele aan als je een andere GPU wilt gebruiken.