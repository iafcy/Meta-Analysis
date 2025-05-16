from typing import Literal

class RubricsColumn():
    def __init__(self, title: str, options: list[str | int], type: Literal["label", "score"]):
        self.title = title
        self.type = type
        self.options = options

        if type == "label":
            optionsStr = "\n".join([f"- {str(option)}" for option in options])
            self.instruction = f"Evaluate the given study and choose the most suitable label from the following:\n{optionsStr}"
        else:
            optionsStr = ", ".join([str(option) for option in options])
            self.instruction = f"Rate the given study and provide the most suitable score from {optionsStr}"

    def to_dict(self):
        return {
            "title": self.title,
            "type": self.type,
            "options": self.options,
            "instruction": self.instruction
        }
