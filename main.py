"""
Ethereum Wallet Risk Scoring

This script analyzes Ethereum wallet activity, focusing on interactions with Compound V2 contracts, and assigns a risk score to each wallet based on on-chain behavior.

Data Collection:
- Transaction data is collected using the Etherscan API for each wallet address.
- Only public on-chain data is used, ensuring transparency and reproducibility.

Feature Selection Rationale:
- Features are chosen to reflect key risk indicators in DeFi lending/borrowing:
    - Borrow count: Frequent borrowing may indicate higher leverage and risk.
    - Repayment ratio: High repayment rates suggest responsible debt management.
    - Collateral ratio: More collateral relative to borrowing reduces liquidation risk.
    - Liquidation count: Past liquidations are a strong risk signal.
    - Total transaction volume: Higher volume may indicate more stable or reputable wallets.
    - Activity level: More active wallets are less likely to be abandoned or malicious.
    - No Compound activity: Wallets with no Compound interactions are mildly penalized.

Scoring Method:
- Features are normalized using MinMaxScaler.
- Each feature is weighted according to its risk relevance (see `weights` in code).
- The weighted sum is scaled to a 0–1000 risk score, where higher scores indicate lower risk.

Justification of Risk Indicators:
- Borrowing without repayment or collateral increases risk of default or liquidation.
- Liquidations are direct evidence of risky or failed positions.
- High activity and volume are generally associated with more reputable or stable users.
- The combination of these indicators provides a holistic risk profile for each wallet.

See README.md for more details.
"""
import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import json
import time
import uuid
import os
from dotenv import load_dotenv

load_dotenv()

# Etherscan API key and Compound V2 contract addresses
API_KEY = os.getenv("Etherscan_API_key")
COMPOUND_CONTRACTS = {
    "cDAI": "0x5d3a536e4d6dbd6114cc1ead35777bab948e3643",
    "cETH": "0x4ddc2d193948926d02f9b1fe9e1daa0718270ed5",
    "cUSDC": "0x39aa39c021dfbae8fac545936693ac917d5e7563",
    "cUSDT": "0xf650c3d88d12be4e4267fcedd83c6e9a4e2c6d5e",
    "Comptroller": "0x3d9819210a31b4961b30ef54be2aed79b9c9cd3b"
}
ETHERSCAN_API_URL = "https://api.etherscan.io/api"

# Wallet addresses from provided CSV
wallet_addresses = [
    "0x0039f22efb07a647557c7c5d17854cfd6d489ef3",
    "0x06b51c6882b27cb05e712185531c1f74996dd988",
    "0x0795732aacc448030ef374374eaae57d2965c16c",
    "0x0aaa79f1a86bc8136cd0d1ca0d51964f4e3766f9",
    "0x0fe383e5abc200055a7f391f94a5f5d1f844b9ae",
    "0x104ae61d8d487ad689969a17807ddc338b445416",
    "0x111c7208a7e2af345d36b6d4aace8740d61a3078",
    "0x124853fecb522c57d9bd5c21231058696ca6d596",
    "0x13b1c8b0e696aff8b4fee742119b549b605f3cbc",
    "0x1656f1886c5ab634ac19568cd571bc72f385fdf7",
    "0x1724e16cb8d0e2aa4d08035bc6b5c56b680a3b22",
    "0x19df3e87f73c4aaf4809295561465b993e102668",
    "0x1ab2ccad4fc97c9968ea87d4435326715be32872",
    "0x1c1b30ca93ef57452d53885d97a74f61daf2bf4f",
    "0x1e43dacdcf863676a6bec8f7d6896d6252fac669",
    "0x22d7510588d90ed5a87e0f838391aaafa707c34b",
    "0x24b3460622d835c56d9a4fe352966b9bdc6c20af",
    "0x26750f1f4277221bdb5f6991473c6ece8c821f9d",
    "0x27f72a000d8e9f324583f3a3491ea66998275b28",
    "0x2844658bf341db96aa247259824f42025e3bcec2",
    "0x2a2fde3e1beb508fcf7c137a1d5965f13a17825e",
    "0x330513970efd9e8dd606275fb4c50378989b3204",
    "0x3361bea43c2f5f963f81ac70f64e6fba1f1d2a97",
    "0x3867d222ba91236ad4d12c31056626f9e798629c",
    "0x3a44be4581137019f83021eeee72b7dc57756069",
    "0x3e69ad05716bdc834db72c4d6d44439a7c8a902b",
    "0x427f2ac5fdf4245e027d767e7c3ac272a1f40a65",
    "0x4814be124d7fe3b240eb46061f7ddfab468fe122",
    "0x4839e666e2baf12a51bf004392b35972eeddeabf",
    "0x4c4d05fe859279c91b074429b5fc451182cec745",
    "0x4d997c89bc659a3e8452038a8101161e7e7e53a7",
    "0x4db0a72edb5ea6c55df929f76e7d5bb14e389860",
    "0x4e61251336c32e4fe6bfd5fab014846599321389",
    "0x4e6e724f4163b24ffc7ffe662b5f6815b18b4210",
    "0x507b6c0d950702f066a9a1bd5e85206f87b065ba",
    "0x54e19653be9d4143b08994906be0e27555e8834d",
    "0x56ba823641bfc317afc8459bf27feed6eb9ff59f",
    "0x56cc2bffcb3f86a30c492f9d1a671a1f744d1d2f",
    "0x578cea5f899b0dfbf05c7fbcfda1a644b2a47787",
    "0x58c2a9099a03750e9842d3e9a7780cdd6aa70b86",
    "0x58d68d4bcf9725e40353379cec92b90332561683",
    "0x5e324b4a564512ea7c93088dba2f8c1bf046a3eb",
    "0x612a3500559be7be7703de6dc397afb541a16f7f",
    "0x623af911f493747c216ad389c7805a37019c662d",
    "0x6a2752a534faacaaa153bffbb973dd84e0e5497b",
    "0x6d69ca3711e504658977367e13c300ab198379f1",
    "0x6e355417f7f56e7927d1cd971f0b5a1e6d538487",
    "0x70c1864282599a762c674dd9d567b37e13bce755",
    "0x70d8e4ab175dfe0eab4e9a7f33e0a2d19f44001e",
    "0x7399dbeebe2f88bc6ac4e3fd7ddb836a4bce322f",
    "0x767055590c73b7d2aaa6219da13807c493f91a20",
    "0x7851bdfb64bbecfb40c030d722a1f147dff5db6a",
    "0x7b4636320daa0bc055368a4f9b9d01bd8ac51877",
    "0x7b57dbe2f2e4912a29754ff3e412ed9507fd8957",
    "0x7be3dfb5b6fcbae542ea85e76cc19916a20f6c1e",
    "0x7de76a449cf60ea3e111ff18b28e516d89532152",
    "0x7e3eab408b9c76a13305ef34606f17c16f7b33cc",
    "0x7f5e6a28afc9fb0aaf4259d4ff69991b88ebea47",
    "0x83ea74c67d393c6894c34c464657bda2183a2f1a",
    "0x8441fecef5cc6f697be2c4fc4a36feacede8df67",
    "0x854a873b8f9bfac36a5eb9c648e285a095a7478d",
    "0x8587d9f794f06d976c2ec1cfd523983b856f5ca9",
    "0x880a0af12da55df1197f41697c1a1b61670ed410",
    "0x8aaece100580b749a20f8ce30338c4e0770b65ed",
    "0x8be38ea2b22b706aef313c2de81f7d179024dd30",
    "0x8d900f213db5205c529aaba5d10e71a0ed2646db",
    "0x91919344c1dad09772d19ad8ad4f1bcd29c51f27",
    "0x93f0891bf71d8abed78e0de0885bd26355bb8b1d",
    "0x96479b087cb8f236a5e2dcbfc50ce63b2f421da6",
    "0x96bb4447a02b95f1d1e85374cffd565eb22ed2f8",
    "0x9a363adc5d382c04d36b09158286328f75672098",
    "0x9ad1331c5b6c5a641acffb32719c66a80c6e1a17",
    "0x9ba0d85f71e145ccf15225e59631e5a883d5d74a",
    "0x9e6ec4e98793970a1307262ba68d37594e58cd78",
    "0xa7e94d933eb0c439dda357f61244a485246e97b8",
    "0xa7f3c74f0255796fd5d3ddcf88db769f7a6bf46a",
    "0xa98dc64bb42575efec7d1e4560c029231ce5da51",
    "0xb271ff7090b39028eb6e711c3f89a3453d5861ee",
    "0xb475576594ae44e1f75f534f993cbb7673e4c8b6",
    "0xb57297c5d02def954794e593db93d0a302e43e5c",
    "0xbd4a00764217c13a246f86db58d74541a0c3972a",
    "0xc179d55f7e00e789915760f7d260a1bf6285278b",
    "0xc22b8e78394ce52e0034609a67ae3c959daa84bc",
    "0xcbbd9fe837a14258286bbf2e182cbc4e4518c5a3",
    "0xcecf5163bb057c1aff4963d9b9a7d2f0bf591710",
    "0xcf0033bf27804640e5339e06443e208db5870dd2",
    "0xd0df53e296c1e3115fccc3d7cdf4ba495e593b56",
    "0xd1a3888fd8f490367c6104e10b4154427c02dd9c",
    "0xd334d18fa6bada9a10f361bae42a019ce88a3c33",
    "0xd9d3930ffa343f5a0eec7606d045d0843d3a02b4",
    "0xdde73df7bd4d704a89ad8421402701b3a460c6e9",
    "0xde92d70252804fd8c5998c8ee3ed282a41b33b7f",
    "0xded1f838ae6aa5fcd0f13481b37ee88e5bdccb3d",
    "0xebb8629e8a3ec86cf90cb7600264415640834483",
    "0xeded1c8c0a0c532195b8432153f3bfa81dba2a90",
    "0xf10fd8921019615a856c1e95c7cd3632de34edc4",
    "0xf340b9f2098f80b86fbc5ede586c319473aa11f3",
    "0xf54f36bca969800fd7d63a68029561309938c09b",
    "0xf60304b534f74977e159b2e159e135475c245526",
    "0xf67e8e5805835465f7eba988259db882ab726800",
    "0xf7aa5d0752cfcd41b0a5945867d619a80c405e52",
    "0xf80a8b9cfff0febf49914c269fb8aead4a22f847",
    "0xfe5a05c0f8b24fca15a7306f6a4ebb7dcf2186ac",
]

def fetch_transactions(wallet_address):
    """Fetch transaction history for a wallet address from Etherscan."""
    params = {
        "module": "account",
        "action": "txlist",
        "address": wallet_address,
        "startblock": 0,
        "endblock": 99999999,
        "sort": "asc",
        "apikey": API_KEY
    }
    response = requests.get(ETHERSCAN_API_URL, params=params)
    time.sleep(0.2)  
    if response.status_code == 200:
        data = response.json()
        if data["status"] == "1":
            return data["result"]
        else:
            print(f"Error fetching data for {wallet_address}: {data['message']}")
            return []
    else:
        print(f"HTTP error for {wallet_address}: {response.status_code}")
        return []

def extract_features(transactions, wallet_address):
    """Extract features from transaction data with improved parsing."""
    borrow_count = 0
    repayment_count = 0
    total_borrowed = 0
    total_repaid = 0
    collateral_deposited = 0
    liquidation_count = 0
    total_volume = 0
    tx_count = 0

    # Common function signatures for Compound V2
    BORROW_SIG = "0x69328dec"  # borrow(uint256)
    REPAY_SIG = "0x0e6798a0"  # repayBorrow(uint256)
    MINT_SIG = "0x1241ab3f"   # mint(uint256)
    LIQUIDATE_SIG = "0x7db4f5c"  # liquidateBorrow(address,uint256,address)

    for tx in transactions:
        value = int(tx["value"]) / 1e18  # Convert Wei to ETH
        total_volume += value
        tx_count += 1
        to_address = tx["to"].lower() if tx["to"] else ""
        input_data = tx["input"].lower()

        # Check for Compound V2 contract interactions
        if to_address in COMPOUND_CONTRACTS.values():
            if input_data.startswith(BORROW_SIG):
                borrow_count += 1
                total_borrowed += value
            elif input_data.startswith(REPAY_SIG):
                repayment_count += 1
                total_repaid += value
            elif input_data.startswith(MINT_SIG):
                collateral_deposited += value
            elif input_data.startswith(LIQUIDATE_SIG):
                liquidation_count += 1

    # Calculate features
    repayment_ratio = total_repaid / (total_borrowed + 1e-10) if total_borrowed > 0 else 1
    collateral_ratio = collateral_deposited / (total_borrowed + 1e-10) if total_borrowed > 0 else 1
    activity_level = tx_count / (tx_count + 1)  # Normalize transaction count

    # Default risk score for wallets with no Compound interactions
    if borrow_count == 0 and repayment_count == 0 and collateral_deposited == 0:
        return {
            "wallet_id": wallet_address,
            "borrow_count": 0,
            "repayment_ratio": 1,
            "collateral_ratio": 1,
            "liquidation_count": 0,
            "total_volume": total_volume,
            "activity_level": activity_level,
            "no_compound_activity": 1
        }
    
    return {
        "wallet_id": wallet_address,
        "borrow_count": borrow_count,
        "repayment_ratio": repayment_ratio,
        "collateral_ratio": collateral_ratio,
        "liquidation_count": liquidation_count,
        "total_volume": total_volume,
        "activity_level": activity_level,
        "no_compound_activity": 0
    }

def calculate_risk_score(features_df):
    """Calculate risk scores for wallets with adjusted logic."""
    # Features to normalize
    feature_columns = ["borrow_count", "repayment_ratio", "collateral_ratio", "liquidation_count", "total_volume", "activity_level", "no_compound_activity"]
    
    # Normalize features to [0, 1]
    scaler = MinMaxScaler()
    normalized_features = scaler.fit_transform(features_df[feature_columns])
    normalized_df = pd.DataFrame(normalized_features, columns=feature_columns, index=features_df.index)
    
    # Adjusted weights to increase score variation
    weights = {
        "borrow_count": -0.2,  # More borrowing increases risk
        "repayment_ratio": 0.3,  # Higher repayment reduces risk
        "collateral_ratio": 0.3,  # Higher collateral reduces risk
        "liquidation_count": -0.25,  # Liquidations increase risk
        "total_volume": 0.2,  # Higher volume indicates stability
        "activity_level": 0.25,  # More transactions indicate engagement
        "no_compound_activity": -0.1  # Mild penalty for no Compound activity
    }
    
    # Calculate weighted score
    scores = np.zeros(len(features_df))
    for feature, weight in weights.items():
        scores += normalized_df[feature] * weight
    
    # Scale scores to [0, 1000] with adjustment to avoid clustering at 0
    min_score, max_score = scores.min(), scores.max()
    if max_score == min_score:  # Avoid division by zero
        scores = np.ones(len(features_df)) * 500  # Default to mid-range score
    else:
        scores = (scores - min_score) / (max_score - min_score) * 900 + 100  # Shift to [100, 1000]
    scores = np.clip(scores, 0, 1000)  # Ensure scores are within [0, 1000]
    features_df["score"] = scores.astype(int)
    return features_df[["wallet_id", "score"]]

def main():
    features_list = []
    for wallet in wallet_addresses:
        print(f"Processing wallet: {wallet}")
        transactions = fetch_transactions(wallet)
        features = extract_features(transactions, wallet)
        features_list.append(features)
    
    # Create DataFrame
    features_df = pd.DataFrame(features_list)
    
    # Calculate risk scores
    result_df = calculate_risk_score(features_df)
    
    # Save to CSV
    result_df.to_csv("wallet_risk_scores.csv", index=False)
    print("Results saved to wallet_risk_scores.csv")

if __name__ == "__main__":
    main()