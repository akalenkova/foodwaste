import pandas as pd
import numpy as np
from scipy.linalg import null_space
import matplotlib.pyplot as plt

# Based on Algorithm 1 from the paper, we now implement the discovery of a CTMC
# from the uploaded Excel data (mean purchase times by quantity for each product).

# Re-load the Excel file (if needed)
file_path = "Mean.xlsx"
df = pd.read_excel(file_path)

# Define a function that implements Algorithm 1
def discover_ctmc_from_product_data(product_df, capacity, initial_quantity):
    """
    Constructs a CTMC (lambda, Q) from product event data as described in Algorithm 1.
    Parameters:
        product_df (DataFrame): Data filtered for a specific product.
        capacity (int): Maximum quantity of the product (state space upper bound).
        initial_quantity (int): Initial quantity in the store (initial state).
    Returns:
        lambda_vec (np.ndarray): Initial state probability vector.
        Q (np.ndarray): Transition rate matrix.
    """
    S = list(range(capacity + 1))  # state space: 0 to k
    Q_matrix = np.zeros((len(S), len(S)))
    lambda_vec = np.zeros(len(S))
    lambda_vec[initial_quantity] = 1  # initial state probability

    for _, row in product_df.iterrows():
        qty = int(row['quantity'])
        mean_hours = row['mean_hours']
        if qty <= capacity and mean_hours > 0:
            for i in range(qty, capacity + 1):
                Q_matrix[i, i - qty] = 1 / mean_hours

    # Fill diagonal: q_ii = -sum(q_ij)
    for i in range(len(S)):
        Q_matrix[i, i] = -np.sum(Q_matrix[i, :])

    return lambda_vec, Q_matrix
def add_supply_transitions(Q_matrix, capacity, supply_batch_size, supply_rate):
    """
    Adds backward supply transitions to the CTMC matrix.
    
    Parameters:
        Q_matrix (np.ndarray): Existing CTMC transition rate matrix.
        capacity (int): Maximum product capacity (number of states - 1).
        supply_batch_size (int): Number of units added in one supply batch.
        supply_rate (float): Rate at which supplies are delivered.
    
    Returns:
        np.ndarray: Enhanced CTMC transition rate matrix with supply transitions.
    """
    Q_enhanced = Q_matrix.copy()
    for i in range(capacity + 1 - supply_batch_size):
        j = i + supply_batch_size
        Q_enhanced[i, j] = supply_rate

    # Update diagonals
    for i in range(capacity + 1):
        Q_enhanced[i, i] = -np.sum(Q_enhanced[i, :]) + Q_enhanced[i, i]  # maintain total row sum = 0

    return Q_enhanced

def compute_steady_state(Q):
    """
    Computes the steady-state probability vector for a CTMC given its Q matrix.
    
    Parameters:
        Q (np.ndarray): CTMC transition rate matrix.
        
    Returns:
        np.ndarray: Steady-state probability vector π such that πQ = 0 and sum(π) = 1.
    """
    # Transpose Q for solving πQ = 0
    QT = Q.T

    # Find the null space (kernel) of Q^T
    ns = null_space(QT)

    # Normalize the result to get a probability vector
    pi = ns[:, 0]
    pi_normalized = pi / np.sum(pi)

    return pi_normalized

# Apply the function to a sample product
sample_product_id = df['product_id'].iloc[0]
product_data = df[df['product_id'] == sample_product_id]
capacity = 100
initial_quantity = 10  

lambda_vec, Q_matrix = discover_ctmc_from_product_data(product_data, capacity, initial_quantity)

# Example usage with batch size 2 and supply rate 0.1 (e.g., one batch every 10 hours)
supply_batch_size = 10
supply_rate = 0.35 # 0.3 per hour
Q_enhanced = add_supply_transitions(Q_matrix, capacity, supply_batch_size, supply_rate)

# Prepare results for display
lambda_series = pd.Series(lambda_vec, name="Initial Probability", index=list(range(capacity + 1)))
q_df = pd.DataFrame(Q_matrix, index=list(range(capacity + 1)), columns=list(range(capacity + 1)))

# Display the results
print("Initial Probability Vector (λ)", lambda_series.to_frame())
print("CTMC Transition Rate Matrix (Q)", q_df)

# Compute steady-state probabilities
steady_state_pi = compute_steady_state(Q_enhanced)

# Display as a DataFrame
steady_state_df = pd.DataFrame({
    "State": list(range(len(steady_state_pi))),
    "Steady-State Probability": steady_state_pi
})

print("Steady-State Probabilities", steady_state_df)

# Plot steady-state probabilities
plt.figure(figsize=(8, 5))
plt.bar(steady_state_df["State"], steady_state_df["Steady-State Probability"], color='#c8a2c8')
plt.xlabel("State (Product Quantity)")
plt.ylabel("Steady-State Probability")
plt.title("Steady-State Distribution of Product Quantity, Supply Rate="+str(supply_rate))
plt.grid(axis='y', linestyle='--', alpha=0.7)
xticks = [x for x in steady_state_df["State"] if x % 5 == 0]
plt.xticks(xticks)
plt.tight_layout()
plt.show()