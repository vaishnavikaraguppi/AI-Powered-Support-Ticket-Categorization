# AI-Powered-Support-Ticket-Categorization

This project uses a local large language model (LLM) to analyze customer support tickets with **zero model fine-tuning**. It performs four tasks:
- Ticket categorization
- Tag generation
- Priority and ETA estimation
- Drafting professional support replies

The goal is to enable faster triage and response for support teams using a low-code AI solution.

---

## Dataset

- `support_tickets.csv` – Contains customer-submitted support queries

---

## Technologies Used

- Python  
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)  
- Hugging Face Hub  
- Pandas  
- Jupyter Notebook

---

## Project Highlights

- Loaded and queried Mistral-7B using `llama-cpp` on CPU  
- Prompted the model with structured tasks:  
  - Categorize issues (e.g., "Login Issue", "Payment Problem")  
  - Suggest relevant tags (max 3)  
  - Estimate priority & time to resolution  
  - Generate complete email replies  
- All responses parsed from JSON or plain text format

---

## Sample Output

| Task            | Sample Output                                  |
|-----------------|------------------------------------------------|
| Category        | `"Technical Bug"`                              |
| Tags            | `["crash", "error", "mobile-app"]`             |
| Priority & ETA  | `{"Priority": "High", "ETA": "24"}`            |
| Response        | *(Full drafted email using model output)*      |

---

##  How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/your-username/support-ticket-analyzer.git
   cd support-ticket-analyzer
