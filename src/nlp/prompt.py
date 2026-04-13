from typing import List, Dict, Any

class PromptBuilder:
    """Builds In-Context Learning prompts for NL2SQL."""
    
    def __init__(self):
        self.instruction = (
            "You are an expert SQL assistant. Given the database schema and a question, "
            "generate the correct SQL query. Only return the SQL query code without explanations."
        )

    def build(self, schema_text: str, question: str, examples: List[Dict[str, Any]] = None) -> str:
        """Constructs a few-shot prompt."""
        prompt_parts = [self.instruction]
        
        # 1. Add Examples (Few-shot)
        if examples:
            prompt_parts.append("\n### Examples:")
            for ex in examples:
                prompt_parts.append(f"Question: {ex['question']}\nSQL: {ex['query']}")
        
        # 2. Add Schema
        prompt_parts.append("\n### Database Schema:")
        prompt_parts.append(schema_text)
        
        # 3. Add current question
        prompt_parts.append("\n### Current Task:")
        prompt_parts.append(f"Question: {question}")
        prompt_parts.append("SQL: ")
        
        return "\n".join(prompt_parts)

    def extract_sql(self, raw_output: str) -> str:
        """Simple cleaner to extract SQL if the model prepends text."""
        # Models sometimes repeat keywords or add markdown blocks
        clean_sql = raw_output.replace("```sql", "").replace("```", "").strip()
        # Take only the first line or first statement if necessary
        return clean_sql.split(';')[0].strip() + ";"
