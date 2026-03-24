# AI-Powered E-commerce Customer Support System

A multi-agent customer support pipeline built with **CrewAI** and **Anthropic Claude** that automatically classifies customer queries, generates empathetic responses, and produces structured ticket reports with escalation decisions.

---

## System Architecture

```
Customer Message (raw text)
         │
         ▼
┌─────────────────────────────────────────────────────┐
│              CrewAI CREW (Sequential)               │
│                                                     │
│  ┌──────────────────────┐                          │
│  │ Intent Classifier    │                          │
│  │ Agent                │                          │
│  │ • Primary intent     │                          │
│  │ • Sentiment level    │                          │
│  │ • Urgency level      │                          │
│  │ • Risk flags         │                          │
│  └──────────┬───────────┘                          │
│             │ intent report                        │
│             ▼                                      │
│  ┌──────────────────────┐                          │
│  │ Support Agent        │                          │
│  │ • Empathetic reply   │                          │
│  │ • Policy reference   │                          │
│  │ • Concrete solution  │                          │
│  └──────────┬───────────┘                          │
│             │ customer response                    │
│             ▼                                      │
│  ┌──────────────────────┐                          │
│  │ Escalation Agent     │                          │
│  │ • Escalation yes/no  │                          │
│  │ • Priority P1-P4     │                          │
│  │ • Ticket report      │                          │
│  └──────────┬───────────┘                          │
└─────────────┼───────────────────────────────────────┘
              │
              ▼
   ┌─────────────────────┐   ┌──────────────────────┐
   │  Customer Response  │   │  Ticket Summary      │
   │  (send to customer) │   │  Report (saved .txt) │
   └─────────────────────┘   └──────────────────────┘
```

---

## Agents

### 1. Intent Classifier Agent
Analyzes the raw customer message and extracts:
- Primary & secondary intent (refund, tracking, damaged item, etc.)
- Sentiment (positive → very angry)
- Urgency level (low → critical)
- Key entities (order numbers, product names, dates)
- Risk flags (chargeback risk, legal threat, social media threat)

### 2. Support Agent
Uses intent analysis to write a customer-facing response:
- Empathetic, warm, professional tone
- Addresses the specific issue with concrete solutions
- References relevant e-commerce policies
- 150-300 words, never robotic

### 3. Escalation Agent
Reviews the full case and generates a ticket report:
- Escalation decision (Yes/No) with reason
- Priority level (P1-Critical to P4-Low)
- Department routing (Refunds, Logistics, Legal, etc.)
- Churn risk assessment
- Notes for human agents

---

## Setup

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate         # Windows CMD

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set API key
# Windows PowerShell:
$env:ANTHROPIC_API_KEY="sk-ant-..."
# Mac/Linux:
export ANTHROPIC_API_KEY="sk-ant-..."

# 4. Run
python main.py
```

---

## Testing Different Scenarios

Edit the last lines of `main.py`:

```python
# Options: "angry_refund", "wrong_item", "tracking_question", "damaged_product"
selected_query = SAMPLE_QUERIES["angry_refund"]
run_support(selected_query)
```

Or pass any custom query:

```python
run_support("I want to return my order #12345, it doesn't fit.")
```

---

## Output

Every run produces:
1. **Customer Response** — printed to terminal, ready to send
2. **Ticket Report** — printed to terminal + saved as `ticket_TIMESTAMP.txt`

### Sample Ticket Report
```
╔══════════════════════════════════════╗
║       CUSTOMER SUPPORT TICKET        ║
╠══════════════════════════════════════╣
TICKET ID       : TKT-20250323-001
DATE & TIME     : 2025-03-23 14:32:11
CHANNEL         : E-commerce Chat
──────────────────────────────────────
CUSTOMER ISSUE  : Undelivered order after 3 weeks
PRIMARY INTENT  : Refund Request
SENTIMENT       : Very Angry
URGENCY         : Critical
RISK FLAGS      : Chargeback Risk, Social Media Threat
──────────────────────────────────────
ESCALATION      : Yes
PRIORITY        : P1 - Critical
DEPARTMENT      : Senior Support + Logistics Team
CHURN RISK      : High
──────────────────────────────────────
RESOLUTION      : Escalated
RESPONSE SENT   : Yes
NOTES           : Customer has contacted 3 times with no response.
                  Immediate human follow-up required within 1 hour.
╚══════════════════════════════════════╝
```

---

## Priority Guide

| Priority | Label    | When to Use |
|----------|----------|-------------|
| P1       | Critical | Legal threats, payment fraud, very angry repeat customer |
| P2       | High     | Damaged goods, wrong items, high-value orders |
| P3       | Medium   | Standard refund/return, shipping delays |
| P4       | Low      | General questions, low-urgency inquiries |

---

## Extending the System

### Add a Resolution Agent
```python
resolution_agent = Agent(
    role="Resolution Specialist",
    goal="Close tickets and send follow-up satisfaction surveys",
    ...
)
resolution_task = Task(..., context=[escalation_task])
```

### Add Multilingual Support
```python
# Add to Support Agent goal:
"Detect the customer's language and respond in the same language."
```

### Connect to a Real Ticketing System
```python
# After run_support(), POST the ticket report to your CRM:
import requests
requests.post("https://your-crm.com/api/tickets", json=result)
```
