# Fantasy Baseball Draft Assistant

A live draft assistant for Yahoo Sports fantasy baseball. Recommends the optimal player to draft each time you're on the clock, using SGP-based valuation, Monte Carlo simulation, and category balance optimization.

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## Setup (Before Draft Day)

### 1. Import Projections

Download CSV files from [FanGraphs Projections](https://www.fangraphs.com/projections) — export hitters and pitchers separately. Upload them in the **Import Data** tab.

You can import from multiple projection systems (Steamer, ZiPS, Depth Charts) and blend them together.

### 2. Import ADP

Download ADP data from [FantasyPros](https://www.fantasypros.com/mlb/adp/overall.php) or similar. Upload in the **Import Data** tab. ADP drives the opponent modeling and survival probability calculations.

### 3. Configure League Settings

In the **League Settings** tab, verify:
- Number of teams (default: 12)
- SGP denominators (defaults are for a typical 12-team 5x5 roto league)
- Risk aversion slider

### 4. (Optional) Connect Yahoo API

In the **Yahoo API** tab, enter your Yahoo Developer credentials to enable:
- Auto-import of league settings (roster slots, categories, team count)
- Live draft pick syncing during the draft

### 5. Validate

In the **Validate & Cheat Sheet** tab:
- Run a benchmark to verify the tool outperforms ADP-based drafting
- Export a printable cheat sheet (CSV or Excel) as a backup
- Check the pre-draft checklist

### 6. Practice

Enable **Practice Mode** when starting a draft to simulate opponent picks automatically. This lets you do a full mock draft using the tool.

## During the Draft

### Layout
Run the app on your laptop. Split-screen: Yahoo draft room on the left, this tool on the right.

### Each Pick
1. **When it's NOT your turn**: Enter each opponent's pick using the search dropdown in the sidebar, or use Yahoo API sync
2. **When it's YOUR turn**: The green banner shows the recommended pick with reasoning. Alternatives are listed below.

### Key Features
- **Big green banner**: Top recommendation with survival probability and category help
- **Alternatives panel**: Next 5 best options with scores and availability risk
- **Position run alerts**: Warns when multiple teams are drafting the same position
- **Category balance**: Visual progress bars showing your strengths and weaknesses
- **Panic mode**: Hit the PANIC button for a simplified, large-text recommendation when time is short
- **Yahoo sync**: Pull picks directly from Yahoo's API instead of manual entry

### Undo
Click **Undo Last Pick** in the sidebar if you entered a pick incorrectly.

### Draft State
Your draft state is auto-saved to `data/draft_state.json`. If the app crashes, relaunch and click **Resume Draft**.

## How It Works

### Valuation Engine
- **SGP (Standings Gain Points)**: Converts raw stats to standings-point movement using league-calibrated denominators
- **Replacement Level**: Computed per position based on league depth
- **VORP**: Value Over Replacement Player — measures how much better a player is than the best freely available alternative
- **Marginal SGP**: Roster-conditional value that accounts for your current category balance
- **Injury Discount**: Optional penalty for injury-prone players (controlled by risk aversion setting)

### Recommendation Scoring
```
pick_score = marginal_sgp + 0.4 * urgency
```
Where urgency is based on survival probability (chance the player is taken before your next pick).

### Opponent Modeling
Opponents are modeled as drafting based on ADP with Gaussian noise (sigma=10 for skilled leagues). Position runs are detected to warn about scarcity.

## Sample Data

To test without real projections:
```bash
python load_sample_data.py
```
This generates ~190 sample players with realistic stat distributions.

## Optional Dependencies

```bash
pip install yfpy        # Yahoo Fantasy API integration
pip install openpyxl    # Excel cheat sheet export
```

## File Structure

```
app.py                  # Streamlit application
src/
  database.py           # SQLite data layer
  valuation.py          # SGP, VORP, category weights
  draft_state.py        # Draft tracking and persistence
  simulation.py         # Monte Carlo simulation, urgency
  yahoo_api.py          # Yahoo Fantasy API client
  validation.py         # Benchmarking and cheat sheets
load_sample_data.py     # Sample data generator
tests/
  profile_latency.py    # Performance profiling
data/                   # SQLite DB, draft state, credentials (gitignored)
```

## League Context

Built for the **FourzynBurn** league on Yahoo Sports:
- 12-team snake draft, 23 rounds
- 5x5 roto: R/HR/RBI/SB/AVG + W/SV/K/ERA/WHIP
- Roster: C, 1B, 2B, 3B, SS, 3 OF, 2 Util, 2 SP, 2 RP, 4 P, 5 BN
