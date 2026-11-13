# Valorant Predictor

How early can you tell whether a VALORANT team is rushing a site, splitting through mid, or setting up a slower execute? This COMP 4107 project explores that question with historical match data. It labels attacking tactics from round events, compares four PyTorch classifiers, and includes a browser-based map viewer for inspecting the underlying plays.

## What it does

- **Imports match data into SQLite.** Parses match JSON into records for matches, players, rounds, kills, and player positions recorded around kill events.
- **Labels attacking tactics.** Uses kill timing, event locations, and spike-plant data to assign one of six classes: A-Site Rush, B-Site Rush, A Split, B Split, Default, and Slow Execute. A split combines early activity in mid and on the target side.
- **Compares static and temporal models.** A multilayer perceptron (MLP) uses a 10-second snapshot. The LSTM, GRU, and Transformer use snapshots at 5-second intervals through 30 seconds. Features include agent roles, A/B/mid presence, player spread, alive counts, and movement.
- **Visualizes match events.** Browse Bind, Breeze, Fracture, Pearl, and Split by match, round, and kill event. The viewer shows player locations, facing directions, and killer and victim markers. It can also show heuristic position estimates generated separately from the tactical classifiers.

## Getting started

Run the commands from the repository root. Use Python 3.10 or newer.

```bash
python -m pip install numpy pandas scikit-learn torch
```

To import matches, put compatible match exports under `json/`. Map subfolders such as `json/Bind/` are supported. The loader and viewer require the project's `map_adjacency_graph/map_zones.py` module. For the viewer's map backgrounds and agent portraits, place images in `map_images/` and `agent_icons/`, respectively.

```bash
python load_data.py
```

Choose **1** to load all JSON files or **2** to select a file. The importer writes to `valorant.db`. If you already have a compatible `valorant.db`, you can start with training or open the viewer directly.

## Train and evaluate

```bash
python train.py --full-pipeline --no-cross-map
```

This labels rounds, writes NumPy arrays under `features/`, and trains all four models using a stratified train/validation/test split. Test metrics include accuracy, balanced accuracy, macro F1, top-2 accuracy, and per-class results. The training script writes its report to `results/training_results.json` and saves the best model checkpoints under `results/`.

Run `python train.py --full-pipeline` to also run leave-one-map-out evaluation and write `results/cross_map_results.json`. If features are already available, run `python train.py --no-cross-map` to train without regenerating them.

### How to interpret the results

The labels are inferred from match events and serve as proxies for team intent. The current feature builder also passes round-level spike-plant fields to early snapshots even when the plant occurs later. Early-round scores therefore reflect an offline experiment and may overstate performance with only the information available at that point in a live round.

## Open the map viewer

```bash
python serve_db.py
```

Open [the local map viewer](http://localhost:8765/valorant_map_viewer_db.html) and select a map, match, and round. To populate the optional position overlays before starting the viewer, run `python attacker_predictions.py` against the imported database.

## Project files

| File | Purpose |
| --- | --- |
| `load_data.py`, `data_loader.py` | Import match JSON and query SQLite records |
| `label_rounds.py` | Apply the tactical labeling rules |
| `features.py` | Build static and temporal feature datasets |
| `models.py`, `train.py` | Define, train, and evaluate the classifiers |
| `attacker_predictions.py` | Estimate player positions for the viewer |
| `serve_db.py`, `valorant_map_viewer_db.html` | Serve the local API and interactive map |
