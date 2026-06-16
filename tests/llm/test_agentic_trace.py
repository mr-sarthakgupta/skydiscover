from skydiscover.llm.agentic_generator import (
    _build_model_inputs,
    _serialize_conversation,
    _serialize_message_content,
)


def test_serialize_message_content_truncates_large_bodies():
    content = "x" * 3000
    serialized = _serialize_message_content(content)
    assert len(serialized) < len(content)
    assert "[3000 chars total]" in serialized


def test_model_inputs_include_system_prompt_and_conversation():
    system_prompt = "Task instructions and agent rules."
    conversation = [
        {"role": "user", "content": "Improve the program."},
        {
            "role": "assistant",
            "content": "Reading files.",
            "tool_calls": [
                {
                    "id": "tc_1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": '{"path": "initial_program.py"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "tc_1", "content": "initial_program.py\n..."},
    ]

    model_inputs = _build_model_inputs(system_prompt, conversation)

    assert model_inputs[0] == {"role": "system", "content": system_prompt}
    assert model_inputs[1:] == _serialize_conversation(conversation)
