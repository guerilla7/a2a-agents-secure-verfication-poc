# A2A Agents Secure Verification POC 
a Python proof-of-concept (PoC) application that outlines how one might approach the formal security verification of LLM Agent applications communicating using the Google A2A (Agent-to-Agent) protocol.

This PoC will simulate the formal verification process by:

1. Modeling the key components of an A2A protocol (Agents, Messages, Tasks, AgentCards).
2. Defining LLM Agents as specialized agents within this model.
3. Establishing a set of security rules and properties derived from research (including OWASP Top 10 for LLMs and A2A security principles).
4. A "verifier" component will then run simulated communication scenarios and apply these rules to identify potential security policy violations or vulnerabilities.

This PoC will not be a full-fledged formal verification tool using mathematical provers (which is a highly complex domain requiring specialized software). Instead, it will demonstrate the structure and type of checks one would consider in such a verification process, tailored to LLM agent interactions over the A2A protocol.

## Key Components:

**1. Models (AgentCard, Agent, LLMAgent, Message, Task, Capability, MessageContentType):**

- These data classes represent the core entities in an A2A communication system.
AgentCard is inspired by the Google A2A concept, holding public metadata about an agent, including its capabilities.
LLMAgent extends Agent with a _simulated_llm_call method. This method is intentionally made vulnerable to a simplistic form of prompt injection for demonstration. It also simulates generating output that might contain sensitive keywords or suggest actions.

**2. AgentRegistry:**

- A simple in-memory registry to store and retrieve AgentCards. In a real system, this could be a distributed service or rely on DNS-based discovery.
For PoC signature verification, it also (unsafely) stores simulated private keys. This is a major simplification for the PoC; real registries would never hold private keys.

**3. A2AProtocolSimulator:**

- Simulates the sending and receiving of messages.
send_message: Simulates signing and (notionally) encrypting messages. It then calls the FormalVerifier before dispatching.
receive_message: Simulates (notionally) decrypting and verifying message signatures. It then passes the payload to the agent (specifically the LLMAgent if applicable) for processing. Crucially, it calls the FormalVerifier upon reception and after the LLM generates output.

**4. FormalVerifier and Security Rules:**

- This is the core of the "formal verification" simulation.
  - log_finding:
      - Records potential security issues.
  - Security Rule Methods (check_rule_...):
  - check_rule_authentication:
      - Checks for message signatures.
  - check_rule_authorization_for_task:
      - Verifies if an agent has the capabilities listed in its AgentCard to perform a given Task.
  - check_rule_confidentiality:
      - Checks if sensitive messages are marked for encryption and if encrypted messages appear (notionally) encrypted.
  - check_rule_integrity:
      - Notes that message integrity relies on signature verification (handled during receive).
  - check_rule_prompt_injection:
      - (OWASP LLM01) A basic check for known textual patterns associated with prompt injection in payloads destined for LLMs.
  - check_rule_llm_input_payload:
      - A wrapper to apply input checks (like prompt injection) to LLM-bound payloads.
  - check_rule_insecure_output_handling:
      - (OWASP LLM02, LLM06) Checks the LLM's generated output for simulated sensitive keywords.
  - check_rule_excessive_agency:
      - (OWASP LLM08) A simplified check to see if an LLM's output suggests an action (e.g., a financial transaction) for which the agent lacks the declared capability.
  - check_all_rules_on_send / check_all_rules_on_receive / check_rule_llm_output_handling:
      - These methods are hooked into the A2AProtocolSimulator at different stages of message processing to apply the relevant rules.

**5. Scenarios:**

- These functions set up specific interaction sequences between agents to test different security aspects:
  - Scenario 1: A baseline secure interaction.
  - Scenario 2: An attempt at prompt injection.
  - Scenario 3: Simulates a message sent without a signature (checking authentication rule on send).
  - Scenario 4: An agent attempts to make another agent perform a task for which the recipient lacks authorization (capability mismatch).
  - Scenario 5: An LLM agent (Bob, without financial capability) receives a prompt that causes its simulated LLM to produce output suggesting a financial action, triggering an "excessive agency" finding.
  - Scenario 6: An LLM agent (Charlie, with financial capability) receives a similar prompt. The "excessive agency" rule for its output related to finance should not trigger, as it's within its declared scope (though the initial prompt injection input rule will still trigger).

### How it Simulates "Formal Verification":

- **Stateful Model**: The Python objects (Agent, Message, etc.) and their attributes represent the state of the system.
- **Transition Simulation**: The A2AProtocolSimulator drives transitions between states (e.g., message sent, message received, LLM processed).
- **Invariant/Property Checking**: The FormalVerifier's rules act as invariants or properties that should hold true. For example, "a sensitive message must always be encrypted" or "an agent must not execute a task it's not authorized for."
-** Violation Detection:** When a rule check fails, a VerificationFinding is logged, indicating a potential vulnerability or policy violation in the modeled scenario.

### To Run This PoC:

1. Save the code as a Python file (e.g., llm_a2a_verifier_poc.py).
2. Run it from your terminal: python llm_a2a_verifier_poc.py
3. Observe the console output, which will show the simulated interactions and the findings from the FormalVerifier.

This PoC provides a tangible starting point for thinking about systematically checking for security issues in complex, multi-agent LLM systems that use structured communication protocols. You can extend it by adding more sophisticated models, rules, and scenarios.
