from abc import abstractmethod
from typing import Dict, Any


class Serializable:
    """可序列化对象的公共基类（协议类，不使用ABC避免与Qt元类冲突）"""
    @abstractmethod
    def serialize(self) -> Dict[str, Any]:
        """将对象序列化为字典，子类必须实现"""
        raise NotImplementedError("子类必须实现 serialize 方法")

    @classmethod
    @abstractmethod
    def deserialize(cls, data: Dict[str, Any], *args, **kwargs):
        """从字典反序列化创建对象，子类必须实现"""
        raise NotImplementedError("子类必须实现 deserialize 方法")
