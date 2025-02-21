from typing import Any, Literal
import datetime

import httpx
from openai import AzureOpenAI
import autogen
from autogen import ChatResult, UserProxyAgent, config_list_from_json
import chromadb
from pydantic import Field, BaseModel, computed_field, model_validator
from rich.console import Console
from autogen.cache import Cache
from rich.markdown import Markdown
from chromadb.config import Settings
from autogen.code_utils import extract_code
from openai.types.completion_usage import CompletionUsage
from openai.types.chat.chat_completion import Choice, ChatCompletion
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from autogen.agentchat.contrib.captainagent.captainagent import CaptainAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
from chromadb.utils.embedding_functions.openai_embedding_function import OpenAIEmbeddingFunction

console = Console()


def get_config_dict(model: str) -> dict[str, Any]:
    config_list = config_list_from_json(
        env_or_file="./configs/OAI_CONFIG_LIST", filter_dict={"model": model}
    )
    llm_config = {
        "timeout": 60,
        "temperature": 0.5,
        "cache_seed": None,
        "config_list": config_list,
    }
    return llm_config


def retrieve_data(query: str) -> str:
    """This tool is for retrieving knowledge from the docs.

    If you have any question, you can give a query keyword then this function will search the answer in the docs and return the answer.

    Args:
        query (str): The question you want to ask and search in the docs.

    Returns:
        str: The answer to the question.

    Examples:
        >>> query = "What is the Product Version of Bandgap Reference Verification"
        >>> retrieved_result = retrieve_data(query=query)
    """
    llm_config = get_config_dict(model="aide-gpt-4o")
    rag_agent = autogen.AssistantAgent(
        name="RetrievalAgent",
        is_termination_msg=lambda msg: "TERMINATE" in msg["content"],
        llm_config=llm_config,
        silent=False,
    )
    rag_user_proxy = RetrieveUserProxyAgent(
        name="RetrieveAgent",
        is_termination_msg=lambda msg: "TERMINATE" in msg["content"],
        human_input_mode="NEVER",
        default_auto_reply="Reply `TERMINATE` if the task is done.",
        max_consecutive_auto_reply=3,
        retrieve_config={
            "task": "default",
            "docs_path": "./docs",
            "must_break_at_empty_line": False,
            "model": "gpt-4",
            "vector_db": None,
            "client": chromadb.Client(Settings(anonymized_telemetry=False)),
            "get_or_create": True,
            "update_context": False,
            "embedding_function": OpenAIEmbeddingFunction(
                api_key=llm_config["config_list"][0]["api_key"],
                api_base="http://mlop-azure-gateway.mediatek.inc",
                api_type=llm_config["config_list"][0]["api_type"],
                api_version=llm_config["config_list"][0]["api_version"],
                model_name="aide-text-embedding-ada-002-v2",
                default_headers=llm_config["config_list"][0]["default_headers"],
            ),
            "embedding_model": None,
        },
        code_execution_config=False,  # we don't want to execute code in this case.
        description="Assistant who has extra content retrieval power for solving difficult problems.",
        silent=False,
    )
    groupchat = autogen.GroupChat(
        agents=[rag_user_proxy, rag_agent],
        messages=[],
        max_round=12,
        speaker_selection_method="round_robin",
    )
    manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config, silent=False)
    retrieve_result = rag_user_proxy.initiate_chat(
        recipient=manager,
        message=rag_user_proxy.message_generator,
        problem=query,
        n_results=1,
        silent=False,
    )
    return retrieve_result.chat_history[-1]["content"]


class ChatResultConverter(BaseModel):
    chat_result: ChatResult = Field(
        ...,
        title="Chat Result from Autogen",
        description="The chat result from autogen groupchat or initiate_chat.",
        frozen=False,
        deprecated=False,
    )
    chat_result_override: str | None = Field(
        default=None,
        title="Chat Result Override",
        description="The real result from the groupchat or captain agent may not contain within the chat_result, so you can override the chat_result with this field by retrieving the messages from the assistant within the groups.",
        frozen=False,
        deprecated=False,
    )
    revert: bool = Field(
        default=True,
        title="Revert the Chat History",
        description="Reverse the chat history, default is True because Analog Coder is using the first message as the result.",
        frozen=True,
        deprecated=False,
    )

    @model_validator(mode="after")
    def _setup(self) -> "ChatResultConverter":
        # HACK: 最後一句話不知道為何有可能是 empty string, 所以用 summary 直接取代以提升兼容性
        # if not self.chat_result.chat_history[-1]["content"]:
        #     self.chat_result.chat_history[-1]["content"] = self.chat_result.summary

        # NOTE: 把對話紀錄翻轉，因為Analog Coder是取第一個當作結果
        if self.revert is True:
            self.chat_result.chat_history = self.chat_result.chat_history[::-1]
        history_alike_summary = [
            {"content": self.chat_result.summary, "role": "assistant", "name": "manual_summary"}
        ]
        # 讓 summary 出現在第一個
        history_alike_summary.extend(self.chat_result.chat_history)
        self.chat_result.chat_history = history_alike_summary
        console.print("+=" * 20, "Chat History Start", "+=" * 20)
        console.print(f"{self.chat_result.chat_history}")
        console.print("+=" * 20, "Chat History End", "+=" * 20)
        return self

    @computed_field
    @property
    def usage(self) -> CompletionUsage:
        cost_dict = self.chat_result.cost["usage_including_cached_inference"]
        if cost_dict:
            for value in cost_dict.values():
                if isinstance(value, dict):
                    completion_tokens = value.get("completion_tokens", 0)
                    prompt_tokens = value.get("prompt_tokens", 0)
                    total_tokens = value.get("total_tokens", 0)
                    result = CompletionUsage(
                        completion_tokens=completion_tokens,
                        prompt_tokens=prompt_tokens,
                        total_tokens=total_tokens,
                    )
                    return result
        return CompletionUsage(completion_tokens=0, prompt_tokens=0, total_tokens=0)

    @computed_field
    @property
    def model(self) -> str:
        cost_dict = self.chat_result.cost["usage_including_cached_inference"]
        if cost_dict:
            for key in cost_dict:
                if key != "total_cost":
                    return key
        return "gpt-4o"

    @computed_field
    @property
    def choices(self) -> list[Choice]:
        choices = []
        if self.chat_result_override:
            pass
        else:
            for idx, chat_dict in enumerate(self.chat_result.chat_history, start=1):
                chat_dict["role"] = "assistant"
                message = ChatCompletionMessage(**chat_dict)
                choice = Choice(finish_reason="stop", index=idx, message=message)
                choices.append(choice)
        return choices

    def convert_to_chat_completion(self) -> ChatCompletion:
        # try:
        chat_completion_result = ChatCompletion(
            id="1",
            choices=self.choices,
            created=int(datetime.datetime.now().timestamp()),
            model=self.model,
            object="chat.completion",
            usage=self.usage,
        )
        # except Exception:
        #     console.print(f"Failed to convert to ChatCompletion:\n{self.chat_result}")
        return chat_completion_result


class CodeBlock(BaseModel):
    code_type: str
    code_content: str

    @computed_field
    @property
    def code_in_markdown(self) -> str:
        return f"```{self.code_type}\n{self.code_content}\n```"


class AnalogAgent(BaseModel):
    use_docker: Literal["mtkomcr.mediatek.inc/srv-aith/mtkllm-sdk-analog", False] = Field(
        default=False,
        title="Use Docker or Not",
        description="Use Docker or Not, if you wanna use, please provide the docker image name.",
        examples=["mtkomcr.mediatek.inc/srv-aith/mtkllm-sdk-analog", False],
    )

    @staticmethod
    def _summary_method(
        sender: autogen.ConversableAgent,
        recipient: CaptainAgent | autogen.ConversableAgent,
        summary_args: dict[str, Any],
    ) -> str:
        if isinstance(recipient, CaptainAgent):
            message_with_codes: list[CodeBlock] = []
            for messages_list in recipient.assistant.chat_messages.values():
                for message in messages_list:
                    message_content = message.get("content")
                    if message_content is None:
                        continue
                    code_messages = extract_code(text=message_content)
                    for code_type, code_content in code_messages:
                        if code_type == "unknown":
                            continue
                        message_with_codes.append(
                            CodeBlock(code_type=code_type, code_content=code_content)
                        )
            if message_with_codes:
                message_with_code = message_with_codes[-1]
                last_message = message_with_code.code_in_markdown
                console.print(
                    "Code Found from the last message, replacing the summary using the following code block:",
                    style="bold green",
                )
                console.print(Markdown(message_with_code.code_in_markdown))
            else:
                last_message = recipient.last_message(sender)["content"]
                console.print("No Code Found from the last message", style="bold red")
        else:
            last_message = recipient.last_message(sender)["content"]
        return last_message

    def _get_cache(self, llm_config: dict[str, Any]) -> Cache:
        cache = None
        cache_seed = llm_config.get("cache_seed")
        if cache_seed:
            cache = Cache.disk(cache_seed=cache_seed, cache_path_root="./.cache")
        return cache

    def use_chat_completion(self, model: str, messages: list[dict[str, str]]) -> ChatCompletion:
        llm_config = get_config_dict(model=model)
        client = AzureOpenAI(
            api_key=llm_config["config_list"][0]["api_key"],
            azure_endpoint=llm_config["config_list"][0]["base_url"],
            api_version=llm_config["config_list"][0]["api_version"],
            http_client=httpx.Client(headers=llm_config["config_list"][0]["default_headers"]),
        )
        result = client.chat.completions.create(messages=messages, model=model, temperature=0.5)
        return result

    def use_groupchat(
        self, model: str, messages: list[dict[str, str]], use_rag: bool
    ) -> ChatCompletion:
        llm_config = get_config_dict(model=model)
        cache = self._get_cache(llm_config=llm_config)
        assistant = autogen.AssistantAgent(
            name="assistant",
            llm_config=llm_config,
            is_termination_msg=lambda x: "TERMINATE" in x.get("content"),
        )
        # create a UserProxyAgent instance named "user_proxy"
        user_proxy = autogen.UserProxyAgent(
            name="user_proxy",
            human_input_mode="NEVER",
            is_termination_msg=lambda x: "TERMINATE" in x.get("content"),
            max_consecutive_auto_reply=10,
            code_execution_config={"use_docker": self.use_docker, "work_dir": "./.cache"},
        )
        if use_rag is True:
            autogen.agentchat.register_function(
                f=retrieve_data,
                caller=assistant,
                executor=user_proxy,
                name=retrieve_data.__name__,
                description=retrieve_data.__doc__,
            )
        chat_result = user_proxy.initiate_chat(
            assistant,
            message=f"{messages}\nPlease make sure the final answer contains the PySpice Code.",
            clear_history=False,
            cache=cache,
        )
        converter = ChatResultConverter(chat_result=chat_result)
        result = converter.convert_to_chat_completion()
        return result

    def use_captain(self, model: str, messages: list[dict[str, str]]) -> ChatCompletion:
        """A Captain Agent by using autogen.

        [ref](https://docs.ag2.ai/docs/use-cases/notebooks/notebooks/agentchat_captainagent).

        Args:
            model (str): The model name you want to use.
            messages (list[dict[str, str]]): The messages you want to chat with the Captain Agent.

        Returns:
            ChatCompletion: The chat result from the Captain Agent.
        """
        llm_config = get_config_dict(model=model)
        cache = self._get_cache(llm_config=llm_config)
        nested_config = {
            "autobuild_init_config": {
                "config_file_or_env": "./configs/OAI_CONFIG_LIST",
                "builder_model": model,
                "agent_model": model,
            },
            "autobuild_build_config": {
                "default_llm_config": {
                    "temperature": 0.5,
                    "top_p": 0.95,
                    "max_tokens": 2048,
                    "cache_seed": llm_config.get("cache_seed"),
                },
                "code_execution_config": {
                    "timeout": 300,
                    "work_dir": "./.cache",
                    "last_n_messages": 1,
                    "use_docker": self.use_docker,
                },
                "coding": True,
            },
            "autobuild_tool_config": {
                # "tool_root": "tools",
                "retriever": "all-mpnet-base-v2"
            },
            "group_chat_config": {"max_round": 10},
            "group_chat_llm_config": None,
            "max_turns": 5,
        }

        captain_user_proxy = UserProxyAgent(
            name="captain_user_proxy",
            llm_config=llm_config,
            human_input_mode="NEVER",
            code_execution_config=False,
            # system_message="Make sure the final answer contains the PySpice Code.",
        )
        captain_agent = CaptainAgent(
            name="captain_agent",
            llm_config=llm_config,
            code_execution_config={
                "timeout": 300,
                "work_dir": "./.cache",
                "last_n_messages": 1,
                "use_docker": self.use_docker,
            },
            agent_config_save_path="./.cache",
            nested_config=nested_config,
            # is_termination_msg=lambda x: "TERMINATE" in x.get("content"),
        )
        chat_result = captain_user_proxy.initiate_chat(
            captain_agent,
            message=f"{messages}\nPlease just seek_experts_help\nPlease make sure the final answer contains the PySpice Code.",
            max_turns=1,
            summary_method=self._summary_method,
            cache=cache,
        )
        converter = ChatResultConverter(chat_result=chat_result)
        result = converter.convert_to_chat_completion()
        return result


def get_chat_completion(
    model: str,
    messages: list[dict[str, str]],
    mode: Literal["original", "captain", "captain+rag", "groupchat", "groupchat+rag"],
    use_docker: Literal["mtkomcr.mediatek.inc/srv-aith/mtkllm-sdk-analog", False] = False,
) -> ChatCompletion:
    analog_agent = AnalogAgent(use_docker=use_docker)
    if mode == "original":
        chat_result = analog_agent.use_chat_completion(model=model, messages=messages)
    elif "captain" in mode:
        if "rag" in mode:
            # chat_result = analog_agent.use_rag_captain(model=model, messages=messages)
            raise NotImplementedError("Not implemented yet.")
        chat_result = analog_agent.use_captain(model=model, messages=messages)
    elif "groupchat" in mode:
        use_rag = False
        if "rag" in mode:
            use_rag = True
        chat_result = analog_agent.use_groupchat(model=model, messages=messages, use_rag=use_rag)
    else:
        raise ValueError(f"Invalid mode: {mode}")
    return chat_result


if __name__ == "__main__":
    model = "aide-gpt-4o"
    messages = [
        {
            "role": "user",
            "content": "Give me a python code that can print a random dataframe; the output should be a python code.",
        }
    ]
    chat_result = get_chat_completion(model=model, messages=messages, mode="original")
    console.print(chat_result)
