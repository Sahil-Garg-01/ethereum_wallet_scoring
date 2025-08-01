**Ethereum Wallet Risk Scoring**

This project analyzes Ethereum wallet activity, especially interactions with Compound V2 contracts, and assigns a risk score to each wallet based on on-chain behavior.

## Features

- Fetches transaction history for a list of Ethereum wallets using the Etherscan API.
- Extracts features such as borrow/repay counts, collateral, liquidations, and transaction volume.
- Calculates a risk score (0–1000) for each wallet using a weighted formula.
- Outputs results to `wallet_risk_scores.csv`.

## Setup

1. **Clone the repository** and navigate to the project directory.

2. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```

3. **Set up your Etherscan API key:**
   - Create a `.env` file in the project root (already present in this repo).
   - Add your API key:
     ```
     Etherscan_API_key=YOUR_ETHERSCAN_API_KEY
     ```

4. **Run the script:**
   ```
   python main.py
   ```

## Files

- `main.py` — Main script for fetching transactions, extracting features, and scoring wallets.
- `.env` — Stores your Etherscan API key (do not share this file publicly).
- `requirements.txt` — Python dependencies.
- `wallet_risk_scores.csv` — Output file with wallet IDs and their risk scores.

## Notes

- The script respects Etherscan API rate limits. If you encounter connection errors, try increasing the delay between requests.
- Risk scoring is based on heuristics and is for demonstration/educational purposes only.

## License

MIT License

