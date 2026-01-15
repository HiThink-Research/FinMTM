import os, base64, json
from openai import OpenAI
import traceback

class Qwen3VLClient:

    def __init__(self, api_base="http://localhost:8000/v1",
                 model="Qwen3-VL-30B-A3B-Instruct",
                 timeout=120):
        self.client = OpenAI(api_key="EMPTY", base_url=api_base)
        self.model = model
        self.timeout = timeout
        self.messages = []  # 用于保存上下文

    # ============ 工具函数 ============
    @staticmethod
    def _to_str(x):
        """确保提示语是字符串"""
        if x is None:
            return ""
        if isinstance(x, str):
            return x
        try:
            return json.dumps(x, ensure_ascii=False)
        except Exception:
            return str(x)

    @staticmethod
    def _image_part(image_path_or_url: str) -> dict:
        """把本地或URL图片转成 OpenAI image_url 部分"""
        if not image_path_or_url:
            print("xx")
            return None

        # 本地文件：转成 base64
        if os.path.exists(image_path_or_url):
            with open(image_path_or_url, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            return {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
            }

        # # 远程 URL
        # return {"type": "image_url", "image_url": {"url": image_path_or_url}}

    # ============ 单轮推理 ============
    def chat(self, image=None, text=None, max_tokens=18000, temperature=0.2):
        """
        单轮对话：支持 image 为单张或多张
        """
        text_str = self._to_str(text)
        content = []

        # 图片可以是字符串或列表
        if image:
            if isinstance(image, str):
                image = [image]
                # print(image)
            else :
                for img in image:
                    part = self._image_part(img)
                    # print(part)
                    if part:
                        content.append(part)

        # 文本
        content.append({"type": "text", "text": text_str})

        # 调用模型
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=self.timeout,
        )
        answer=resp.choices[0].message.content.strip()
        refined=answer.split("</think>", 1)[-1]

        return refined

    # ============ 多轮对话 ============
    def chat_with_memory(self, text=None, image=None,messages=None,
                         max_tokens=18000, temperature=0.2):
        """
        多轮上下文对话：内部自动维护 messages
        """
        text_str = self._to_str(text)
        content = []
        print("image:",image)
        if image:
            if isinstance(image, str):
                image = [image]
                # print(image)
            for img in image:
                part = self._image_part(img)
                # print(part)
                if part:
                    content.append(part)

        # 支持多图输入
        # if image:
        #     if isinstance(image, str):
        #         image = [image]
        #     for img in image:
        #         part = self._image_part(img)
        #         # print("part:",part)
        #         if part:
        #             print("add")
        #             content.append(part)

        content.append({"type": "text", "text": text_str})
        # print(content)
        messages.append({"role": "user", "content": content})
        # print(messages)

        # 调用模型
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=self.timeout,
        )

        answer = resp.choices[0].message.content.strip()
       
        refined=answer.split("</think>", 1)[-1]
        answer=refined
        print(answer)
        # self.messages.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
        return answer

    def get_embedding(self, text, max_tokens=18000, temperature=0.2):
        """
        调用 embeddings 接口获取向量
        """
        content=[]
        text = self._to_str(text)
        content.append({"type": "text", "text": text})
        # messages.append({"role": "user", "content": content})
        
        try:
            resp = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            print(resp)
            # 2. 提取向量数据
            # 只有 embeddings 接口的返回结果里才有 .data
            return resp.data[0].embedding
        except Exception as e:
            print(f"[Error] 获取Embedding失败: {e}")
            return []
    # ============ 清空上下文 ============
    def clear(self):
        """清空上下文"""
        self.messages = []

