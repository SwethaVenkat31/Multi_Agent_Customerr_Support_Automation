"""
AI-Powered Multi-Agent Customer Support System — E-commerce Edition
====================================================================
Agents:
  1. Intent Classifier Agent  — understands what the customer wants
  2. Support Agent            — answers the customer's question
  3. Escalation Agent         — handles complex/angry cases & decides escalation

Output:
  - Customer-facing response
  - Ticket summary report (intent, severity, resolution, escalation decision)

Usage:
  python main.py
  OR: from main import run_support(customer_query)
"""

import os
from datetime import datetime
from crewai import Agent, Task, Crew, Process, LLM


# ---------------------------------------------------------------------------
# 1. LLM Configuration
# ---------------------------------------------------------------------------

def get_llm() -> LLM:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY is not set.\n"
            "Windows PowerShell: $env:ANTHROPIC_API_KEY='sk-ant-...'\n"
            "Mac/Linux:          export ANTHROPIC_API_KEY='sk-ant-...'"
        )
    return LLM(
        model="anthropic/claude-sonnet-4-5",
        api_key=api_key,
        temperature=0.4,   # lower temp for consistent support responses
        max_tokens=4096,
    )


# ---------------------------------------------------------------------------
# 2. Agent Definitions
# ---------------------------------------------------------------------------

def create_intent_classifier_agent(llm: LLM) -> Agent:
    """
    Intent Classifier Agent
    -----------------------
    Reads the raw customer message and identifies:
    - Intent category (refund, tracking, product issue, complaint, etc.)
    - Sentiment (positive, neutral, frustrated, angry)
    - Urgency level (low, medium, high, critical)
    - Key entities (order number, product name, etc.)
    """
    return Agent(
        role="Customer Intent Analyst",
        goal=(
            "Accurately analyze customer messages to identify their intent, "
            "emotional sentiment, urgency level, and key details like order numbers "
            "or product names. Provide a structured classification that guides "
            "the support and escalation agents."
        ),
        backstory=(
            "You are a specialist in Natural Language Understanding with 10 years "
            "of experience in e-commerce customer behavior analysis. You can instantly "
            "detect what a customer truly needs — even when they're frustrated or unclear "
            "— and you categorize issues with precision so the right team handles them."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )


def create_support_agent(llm: LLM) -> Agent:
    """
    Support Agent
    -------------
    Uses the classified intent to craft a helpful, empathetic, and accurate
    response tailored to the customer's specific e-commerce issue.
    """
    return Agent(
        role="Senior E-commerce Support Specialist",
        goal=(
            "Provide clear, empathetic, and accurate responses to customer queries "
            "about orders, products, refunds, shipping, returns, and account issues. "
            "Always maintain a warm, professional tone and offer concrete next steps."
        ),
        backstory=(
            "You are a seasoned e-commerce support specialist with 8 years of experience "
            "handling thousands of customer issues at top online retail companies. "
            "You know every policy — refunds, returns, shipping delays, damaged goods — "
            "and you communicate solutions in a way that leaves customers feeling heard "
            "and satisfied. You always offer actionable solutions, not vague reassurances."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )


def create_escalation_agent(llm: LLM) -> Agent:
    """
    Escalation Agent
    ----------------
    Reviews the intent analysis and support response to decide:
    - Whether the case needs human escalation
    - What priority level to assign
    - Generates a complete ticket summary report
    """
    return Agent(
        role="Customer Experience Escalation Manager",
        goal=(
            "Review customer cases thoroughly to determine if human escalation is needed. "
            "Assign appropriate priority levels, identify risk factors (churn risk, legal risk, "
            "social media risk), and generate a complete structured ticket summary report "
            "for the support team."
        ),
        backstory=(
            "You are a senior customer experience manager with 12 years in e-commerce "
            "escalation handling. You have a sharp eye for cases that could spiral into "
            "chargebacks, negative reviews, or legal complaints. You make data-driven "
            "escalation decisions and produce crisp, actionable ticket reports that "
            "help human agents resolve issues quickly when they take over."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )


# ---------------------------------------------------------------------------
# 3. Task Definitions
# ---------------------------------------------------------------------------

def create_intent_task(agent: Agent, customer_query: str) -> Task:
    return Task(
        description=(
            f"Analyze the following customer message from our e-commerce platform:\n\n"
            f"---\n{customer_query}\n---\n\n"
            f"Identify and extract:\n"
            f"1. Primary Intent (e.g. refund request, order tracking, product complaint, "
            f"   return request, account issue, delivery problem, wrong item, damaged item)\n"
            f"2. Secondary Intent (if any)\n"
            f"3. Customer Sentiment: positive / neutral / frustrated / angry / very angry\n"
            f"4. Urgency Level: low / medium / high / critical\n"
            f"5. Key Entities: order numbers, product names, dates mentioned\n"
            f"6. Potential Risk Flags: chargeback risk, legal threat, social media threat, "
            f"   repeat complaint\n"
            f"7. Recommended Handling: self-service / standard support / priority support / "
            f"   immediate escalation"
        ),
        expected_output=(
            "A structured intent analysis report containing:\n"
            "- PRIMARY INTENT: [category]\n"
            "- SECONDARY INTENT: [category or None]\n"
            "- SENTIMENT: [level] — with 1-sentence explanation\n"
            "- URGENCY: [level] — with reason\n"
            "- KEY ENTITIES: [list of order numbers, products, dates]\n"
            "- RISK FLAGS: [list of risks or None]\n"
            "- RECOMMENDED HANDLING: [approach]\n"
            "- SUMMARY: 2-3 sentence plain English summary of the customer's situation"
        ),
        agent=agent,
    )


def create_support_task(agent: Agent, intent_task: Task, customer_query: str) -> Task:
    return Task(
        description=(
            f"Using the intent analysis provided, craft a complete customer support response "
            f"for this e-commerce customer query:\n\n"
            f"---\n{customer_query}\n---\n\n"
            f"Your response must:\n"
            f"1. Open with empathy — acknowledge the customer's frustration or concern\n"
            f"2. Address their PRIMARY intent directly with a concrete solution or next step\n"
            f"3. Provide specific policy information relevant to their issue "
            f"   (e.g. 30-day return policy, 3-5 day refund processing time)\n"
            f"4. Offer at least ONE additional helpful action or tip\n"
            f"5. Close warmly with a clear call to action\n"
            f"6. Keep the tone: professional, warm, and human — never robotic\n\n"
            f"E-commerce policies to reference when relevant:\n"
            f"- Returns accepted within 30 days of delivery\n"
            f"- Refunds processed in 3-5 business days\n"
            f"- Free replacement for damaged/wrong items\n"
            f"- Order tracking available via account dashboard\n"
            f"- Standard shipping: 5-7 days | Express: 2-3 days | Same-day in select cities"
        ),
        expected_output=(
            "A complete customer-facing support response that:\n"
            "- Starts with a warm, empathetic greeting\n"
            "- Directly addresses the customer's issue\n"
            "- Includes specific next steps or solutions\n"
            "- References relevant policy when applicable\n"
            "- Ends with a friendly closing and contact offer\n"
            "- Is 150-300 words in length\n"
            "- Uses a warm, professional, human tone throughout"
        ),
        agent=agent,
        context=[intent_task],
    )


def create_escalation_task(
    agent: Agent,
    intent_task: Task,
    support_task: Task,
    customer_query: str,
) -> Task:
    return Task(
        description=(
            f"Review the complete customer case below and make an escalation decision. "
            f"Then generate a full ticket summary report.\n\n"
            f"Original customer query: {customer_query}\n\n"
            f"Use the intent analysis and support response provided in context.\n\n"
            f"Your tasks:\n"
            f"1. Decide: Does this case need human agent escalation? (Yes/No + reason)\n"
            f"2. Assign ticket priority: P1-Critical / P2-High / P3-Medium / P4-Low\n"
            f"3. Identify escalation department if needed: "
            f"   Refunds Team / Logistics Team / Legal Team / Senior Support / "
            f"   Social Media Team / None\n"
            f"4. Assess churn risk: High / Medium / Low\n"
            f"5. Generate complete ticket summary report\n\n"
            f"Priority Guidelines:\n"
            f"- P1: Legal threats, payment fraud, safety issues, very angry repeat customer\n"
            f"- P2: High-value order issues, damaged goods, wrong items, angry customer\n"
            f"- P3: Standard refund/return requests, shipping delays, product questions\n"
            f"- P4: General inquiries, positive feedback, low-urgency questions"
        ),
        expected_output=(
            "A complete ticket summary report in this exact format:\n\n"
            "╔══════════════════════════════════════╗\n"
            "║       CUSTOMER SUPPORT TICKET        ║\n"
            "╠══════════════════════════════════════╣\n"
            "TICKET ID       : [auto-generated]\n"
            "DATE & TIME     : [current datetime]\n"
            "CHANNEL         : E-commerce Chat\n"
            "──────────────────────────────────────\n"
            "CUSTOMER ISSUE  : [1-line summary]\n"
            "PRIMARY INTENT  : [intent category]\n"
            "SENTIMENT       : [sentiment level]\n"
            "URGENCY         : [urgency level]\n"
            "RISK FLAGS      : [risks or None]\n"
            "──────────────────────────────────────\n"
            "ESCALATION      : [Yes/No]\n"
            "PRIORITY        : [P1/P2/P3/P4 - label]\n"
            "DEPARTMENT      : [department or None]\n"
            "CHURN RISK      : [High/Medium/Low]\n"
            "──────────────────────────────────────\n"
            "RESOLUTION      : [Resolved/Pending/Escalated]\n"
            "RESPONSE SENT   : [Yes/No]\n"
            "NOTES           : [2-3 sentences for human agent]\n"
            "╚══════════════════════════════════════╝"
        ),
        agent=agent,
        context=[intent_task, support_task],
    )


# ---------------------------------------------------------------------------
# 4. Crew Setup and Execution
# ---------------------------------------------------------------------------

def build_crew(customer_query: str) -> Crew:
    llm = get_llm()

    # Agents
    classifier = create_intent_classifier_agent(llm)
    support    = create_support_agent(llm)
    escalation = create_escalation_agent(llm)

    # Tasks
    intent_task    = create_intent_task(classifier, customer_query)
    support_task   = create_support_task(support, intent_task, customer_query)
    escalation_task = create_escalation_task(
        escalation, intent_task, support_task, customer_query
    )

    return Crew(
        agents=[classifier, support, escalation],
        tasks=[intent_task, support_task, escalation_task],
        process=Process.sequential,
        verbose=True,
    )


def run_support(customer_query: str) -> dict:
    """
    Main entry point. Runs the full support pipeline.

    Args:
        customer_query: Raw customer message/complaint

    Returns:
        dict with keys: 'response', 'ticket_report', 'full_output'
    """
    print(f"\n{'='*60}")
    print(f"  E-commerce Customer Support AI System")
    print(f"{'='*60}")
    print(f"  Customer Query: {customer_query[:80]}...")
    print(f"{'='*60}\n")

    crew   = build_crew(customer_query)
    result = crew.kickoff()

    full_output = result.raw if hasattr(result, "raw") else str(result)

    # Extract ticket report (last section of output)
    ticket_report = ""
    response_text = ""
    if "╔" in full_output:
        parts = full_output.split("╔")
        response_text = parts[0].strip()
        ticket_report = "╔" + parts[1] if len(parts) > 1 else full_output
    else:
        response_text = full_output
        ticket_report = "Ticket report not generated separately."

    print(f"\n{'='*60}")
    print("  CUSTOMER RESPONSE")
    print(f"{'='*60}")
    print(response_text)

    print(f"\n{'='*60}")
    print("  TICKET SUMMARY REPORT")
    print(f"{'='*60}")
    print(ticket_report)

    # Auto-save ticket report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"ticket_{timestamp}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"CUSTOMER QUERY:\n{customer_query}\n\n")
        f.write(f"SUPPORT RESPONSE:\n{response_text}\n\n")
        f.write(f"TICKET REPORT:\n{ticket_report}\n")
    print(f"\n  Ticket saved to: {filename}")

    return {
        "response":      response_text,
        "ticket_report": ticket_report,
        "full_output":   full_output,
    }


# ---------------------------------------------------------------------------
# 5. Sample Customer Queries — change these to test different scenarios
# ---------------------------------------------------------------------------

SAMPLE_QUERIES = {
    "angry_refund": (
        "I ordered a laptop bag 3 weeks ago (Order #ORD-58291) and it still hasn't arrived! "
        "I've sent 3 emails and nobody replied. This is absolutely unacceptable. "
        "I want a full refund immediately or I'm disputing this with my bank and "
        "posting about this on Twitter. Worst online shopping experience ever."
    ),
    "wrong_item": (
        "Hi, I received my order today (Order #ORD-44821) but I got the wrong color. "
        "I ordered the blue sneakers in size 9 but received red ones in size 10. "
        "Can you help me get the correct item?"
    ),
    "tracking_question": (
        "Hello, I placed an order 4 days ago (Order #ORD-67123) and I haven't received "
        "any tracking information yet. Could you please tell me the status of my order? "
        "I need it by Friday for a special occasion."
    ),
    "damaged_product": (
        "The phone case I ordered arrived completely shattered. The packaging was also "
        "damaged so it was clearly mishandled during shipping. Order #ORD-99012. "
        "I need a replacement urgently as it was a gift."
    ),
}

if __name__ == "__main__":
    # Change the key to test different customer scenarios:
    # Options: "angry_refund", "wrong_item", "tracking_question", "damaged_product"
    selected_query = SAMPLE_QUERIES["angry_refund"]
    run_support(selected_query)
