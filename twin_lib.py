from langchain_ollama import ChatOllama
from typing import Optional


class Agent:
    """Individual agent that can discuss and form opinions on questions."""
    
    def __init__(self, name: str, model_name: str, ollama_url: Optional[str] = None, temperature: float = 0.7):
        self.name = name
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.temperature = temperature
        self.llm = ChatOllama(
            model=self.model_name,
            temperature=self.temperature,
            base_url=self.ollama_url,
            num_ctx=6140
        )
        self.conversation_history = []
    
    def get_initial_response(self, question: str) -> str:
        """Get the agent's initial response to a question."""
        messages = [
            ("system", f"You are {self.name}, a thoughtful and knowledgeable assistant. "
                      "Provide a clear, concise answer to the question."),
            ("user", question)
        ]
        self.conversation_history = messages.copy()
        response = self.llm.invoke(messages)
        answer = response.content
        self.conversation_history.append(("assistant", answer))
        return answer
    
    def respond_to_other_agent(self, other_agent_name: str, other_agent_response: str, original_question: str) -> str:
        """Respond to what another agent said."""
        user_prompt = (
            f"{other_agent_name} said:\n\"{other_agent_response}\"\n\n"
            f"Do you agree or disagree? Provide your response to their answer about: {original_question}"
        )
        
        system_prompt = (
            f"You are {self.name}, a thoughtful assistant participating in a discussion. "
            "Keep responses concise and focused. If you agree with the other agent's point, "
            "say so clearly. If you disagree, explain your reasoning."
        )
        
        messages = self.conversation_history.copy()
        messages.append(("system", system_prompt))
        messages.append(("user", user_prompt))
        
        response = self.llm.invoke(messages)
        answer = response.content
        
        # Update history
        self.conversation_history.append(("user", user_prompt))
        self.conversation_history.append(("assistant", answer))
        
        return answer
    
    def check_agreement(self, other_agent_response: str) -> bool:
        """Check if this agent agrees with the other agent's response."""
        check_prompt = (
            f"Based on our discussion, do you fundamentally agree with the other agent's conclusion? "
            f"Respond with only 'YES' or 'NO'."
        )
        
        messages = self.conversation_history.copy()
        messages.append(("user", check_prompt))
        
        response = self.llm.invoke(messages)
        answer = response.content.strip().upper()
        
        return "YES" in answer


class TwinAgentDiscussion:
    """Two agents that discuss a question until they reach agreement."""
    
    def __init__(
        self,
        model_name: str,
        ollama_url: Optional[str] = None,
        temperature: float = 0.7,
        max_rounds: int = 5
    ):
        """
        Initialize the twin agent discussion system.
        
        Args:
            model_name: Name of the model to use (must be available in Ollama)
            ollama_url: URL of the Ollama server (default: None, uses Ollama default)
            temperature: Temperature for model responses (default: 0.7)
            max_rounds: Maximum number of discussion rounds before stopping (default: 5)
        """
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.temperature = temperature
        self.max_rounds = max_rounds
        
        self.agent1 = Agent("Agent-Alpha", model_name, ollama_url, temperature)
        self.agent2 = Agent("Agent-Beta", model_name, ollama_url, temperature)
        self.discussion_log = []
    
    def _log_message(self, agent_name: str, message: str):
        """Log a message to the discussion log."""
        self.discussion_log.append(f"[{agent_name}]: {message}")
    
    def get_answer(self, question: str) -> str:
        """
        Get an answer to a question through agent discussion.
        
        Args:
            question: The question to answer
            
        Returns:
            The final agreed-upon answer
        """
        print(f"\n{'='*60}")
        print(f"Question: {question}")
        print(f"{'='*60}\n")
        
        # Round 0: Get initial responses
        print("Round 0: Initial responses")
        response1 = self.agent1.get_initial_response(question)
        self._log_message("Agent-Alpha", response1)
        print(f"Agent-Alpha: {response1[:100]}...")
        
        response2 = self.agent2.get_initial_response(question)
        self._log_message("Agent-Beta", response2)
        print(f"Agent-Beta: {response2[:100]}...")
        
        current_response1 = response1
        current_response2 = response2
        
        # Discussion rounds
        for round_num in range(1, self.max_rounds + 1):
            print(f"\nRound {round_num}: Discussing")
            
            # Agent 1 responds to Agent 2
            response1 = self.agent1.respond_to_other_agent("Agent-Beta", current_response2, question)
            self._log_message("Agent-Alpha", response1)
            print(f"Agent-Alpha: {response1[:100]}...")
            
            # Agent 2 responds to Agent 1
            response2 = self.agent2.respond_to_other_agent("Agent-Alpha", response1, question)
            self._log_message("Agent-Beta", response2)
            print(f"Agent-Beta: {response2[:100]}...")
            
            # Check for agreement
            agent1_agrees = self.agent1.check_agreement(response2)
            agent2_agrees = self.agent2.check_agreement(response1)
            
            print(f"Agent-Alpha agrees: {agent1_agrees}, Agent-Beta agrees: {agent2_agrees}")
            
            if agent1_agrees and agent2_agrees:
                print(f"\n✓ Agents reached agreement after {round_num} discussion round(s)!")
                break
            
            current_response1 = response1
            current_response2 = response2
        
        # Generate final answer
        final_answer = self._generate_consensus_answer(question, current_response1, current_response2)
        
        return final_answer
    
    def _generate_consensus_answer(self, question: str, response1: str, response2: str) -> str:
        """Generate a consensus answer from both agents' final responses."""
        consensus_prompt = (
            f"Based on the following two responses to the question: \"{question}\"\n\n"
            f"Agent-Alpha's answer:\n{response1}\n\n"
            f"Agent-Beta's answer:\n{response2}\n\n"
            f"Please provide a final, consolidated answer that captures the consensus. "
            f"If there are differences, synthesize them into one coherent answer."
        )
        
        messages = [
            ("system", "You are a consensus builder. Synthesize two perspectives into one clear answer."),
            ("user", consensus_prompt)
        ]
        
        response = self.agent1.llm.invoke(messages)
        final_answer = response.content
        
        print(f"\n{'='*60}")
        print("Final Consensus Answer:")
        print(f"{'='*60}")
        print(final_answer)
        print(f"{'='*60}\n")
        
        return final_answer
    
    def get_discussion_log(self) -> list:
        """Get the complete discussion log."""
        return self.discussion_log


def ask_question(
    question: str,
    model_name: str,
    ollama_url: Optional[str] = None,
    temperature: float = 0.7,
    max_rounds: int = 5
) -> str:
    """
    Ask a question and get an answer from two discussing agents.
    
    Args:
        question: The question to answer
        model_name: Name of the Ollama model to use
        ollama_url: URL of the Ollama server (default: None, uses Ollama default)
        temperature: Temperature for model responses (default: 0.7)
        max_rounds: Maximum number of discussion rounds (default: 5)
        
    Returns:
        The final agreed-upon answer to the question
        
    Example:
        >>> answer = ask_question(
        ...     "What is the capital of France?",
        ...     model_name="mistral",
        ...     ollama_url="http://localhost:11434"
        ... )
    """
    discussion = TwinAgentDiscussion(
        model_name=model_name,
        ollama_url=ollama_url,
        temperature=temperature,
        max_rounds=max_rounds
    )
    
    return discussion.get_answer(question)
