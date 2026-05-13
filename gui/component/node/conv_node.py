from typing import Dict, Any
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QWidget, QHBoxLayout, QLabel, QSpinBox, QCheckBox

from gui.component.node.base_node import BaseNode
from gui.component.socket import Socket
from gui.utils.common import SocketData, FONT_DATA

temptemp = 0
maxcal = 0

class ConvNode(BaseNode):
    panel = None

    def __init__(self, name: str, view_name: str, input_size: int, output_size: int, is_csm: bool = False, is_DW: bool = False, default_depth: int = 1):
        self.father = None
        self.is_DW = is_DW
        self.is_csm = is_csm
        self.parallelism = 1
        self.input_size = input_size
        self.output_size = output_size
        super().__init__(name, view_name, default_depth, [SocketData("input", input_size)], [SocketData("output", output_size)])
        self.is_bypass = False

    def gen_channel_code(self, base_name, depth_val, output):
        if depth_val <= 0:
            depth_val = self.default_depth
        if not self.is_csm:
            pass
        else:
            if self.is_bypass:
                output.append(f"channel int8_t {base_name}[2] __attribute__((depth({depth_val})));")
                self.out_sockets[0].edges[0].base_name = base_name
                self.out_sockets[0].edges[0].extra_name = "[0]"
                self.out_sockets[1].edges[0].base_name = base_name
                self.out_sockets[1].edges[0].extra_name = "[1]"
            else:
                output.append(
                    f"channel int8_t {base_name} __attribute__((depth({depth_val})));")
                self.out_sockets[0].edges[0].base_name = base_name

    def gen_node_code(self, output):
        if not self.is_csm:
            pass
        else:

            global temptemp
            if self.is_DW:
                temptemp += 2 * (self.father.kernel_size * self.father.kernel_size) * self.father.output_size * self.father.output_size * self.father.output_channels
            else:
                temptemp += 2 * (self.father.kernel_size * self.father.kernel_size * self.father.input_channels) * self.father.output_size * self.father.output_size * self.father.output_channels
            print(temptemp)

            temp = self.parallelism * 8 // 8

            if temp < 1:
                temp = 1

            coarse = temp

            # global maxcal
            # if self.is_DW:
            #     temptemp =  self.father.output_size * self.father.output_size * self.father.output_channels // coarse
            # else:
            #     temptemp =   self.father.input_channels * self.father.output_size * self.father.output_size * self.father.output_channels // coarse
            # if temptemp > maxcal:
            #     maxcal = temptemp
            #
            # print(self.view_name)
            # print(temptemp)
            # print(maxcal)

            if self.is_bypass:
                input_name = self.in_sockets[0].edges[0].get_base_name()
                output_name = self.out_sockets[0].edges[0].get_base_name()
                if self.is_DW:
                    macro = "GEN_PIPE_DWCSM_LAYER"
                else:
                    macro = "GEN_PIPE_CSM_LAYER"
                output.append(f"#define COARSE_{self.name.upper()} {coarse}")
                output.append(
                    f"{macro}({self.name}, {input_name}, {output_name}, COARSE_{self.name.upper()}, int8_t, int)")
            else:
                input_name = self.in_sockets[0].edges[0].get_total_name()
                output_name = self.out_sockets[0].edges[0].get_total_name()
                if self.is_DW:
                    macro = "GEN_DWCSM_LAYER"
                else:
                    macro = "GEN_CSM_LAYER"
                output.append(f"#define COARSE_{self.name.upper()} {coarse}")
                output.append(
                    f"{macro}({self.name}, {input_name}, {output_name}, COARSE_{self.name.upper()}, int8_t, int)")

    def set_parallelism(self, p):
        self.parallelism = p

    def set_bypass(self, state):
        self.is_bypass = state
        if self.is_bypass:
            if len(self.in_sockets) == 1:
                new_sock = Socket(self, SocketData("input_bypass", self.input_size), True)
                self.in_sockets.append(new_sock)
                if self.scene(): self.scene().addItem(new_sock)
            if len(self.out_sockets) == 1:
                new_sock = Socket(self, SocketData("output_bypass", self.output_size), False)
                self.out_sockets.append(new_sock)
                if self.scene(): self.scene().addItem(new_sock)
        else:
            if len(self.in_sockets) > 1:
                sock = self.in_sockets.pop()
                for e in sock.edges[:]: e.remove_self()
                if self.scene(): self.scene().removeItem(sock)
            if len(self.out_sockets) > 1:
                sock = self.out_sockets.pop()
                for e in sock.edges[:]: e.remove_self()
                if self.scene(): self.scene().removeItem(sock)

        self._arrange_sockets()
        self.update()

    def paint(self, painter, option, widget):
        super().paint(painter, option, widget)
        painter.setFont(FONT_DATA)
        painter.setPen(QColor("#f1c40f"))
        painter.drawText(0, 60, self.width, 30, Qt.AlignmentFlag.AlignCenter, f"P: {self.parallelism}")
        if self.is_bypass:
            painter.drawText(0, 85, self.width, 30, Qt.AlignmentFlag.AlignCenter, "[BYPASS]")

    @classmethod
    def init_widget(cls, panel):
        cls.panel = panel
        m_name = cls.__name__
        widget_map = panel.widget_map
        widget_map.setdefault(m_name, {})

        widget_map[m_name]["p_widget"] = QWidget()
        p_widget = widget_map[m_name]["p_widget"]
        p_lay = QHBoxLayout(p_widget)
        p_lay.setContentsMargins(0, 0, 0, 0)
        p_lay.addWidget(QLabel("并行度:"))
        widget_map[m_name]["p_spin"] = QSpinBox()
        p_spin = widget_map[m_name]["p_spin"]
        p_spin.setRange(1, 2147483647)
        p_spin.valueChanged.connect(cls.on_p_changed)
        p_lay.addWidget(p_spin)
        panel.layout.addWidget(p_widget)
        p_widget.hide()

        widget_map[m_name]["bypass_check"] = QCheckBox("启用旁路 (Bypass)")
        bypass_check = widget_map[m_name]["bypass_check"]
        bypass_check.stateChanged.connect(cls.on_bypass_changed)
        panel.layout.addWidget(bypass_check)
        bypass_check.hide()

    @classmethod
    def hide_widget(cls, item):
        m_name = cls.__name__
        widget_map = cls.panel.widget_map
        widget_map[m_name]["p_widget"].hide()
        widget_map[m_name]["bypass_check"].hide()

    def set_widget_value(self, item, panel):
        m_name = self.__class__.__name__

        panel.add_info_row("is_csm", self.is_csm)
        panel.add_info_row("is_DW", self.is_DW)

        widget_map = self.panel.widget_map
        p_widget = widget_map[m_name]["p_widget"]
        p_spin = widget_map[m_name]["p_spin"]
        bypass_check = widget_map[m_name]["bypass_check"]

        p_widget.show()
        bypass_check.show()
        p_spin.setValue(item.parallelism)
        bypass_check.setChecked(item.is_bypass)

    @classmethod
    def on_p_changed(cls, val):
        if isinstance(cls.panel.current_item, ConvNode):
            cls.panel.current_item.set_parallelism(val)
            # self.current_item.update()

    @classmethod
    def on_bypass_changed(cls, state):
        if isinstance(cls.panel.current_item, ConvNode):
            cls.panel.current_item.set_bypass(state == 2)

    # ========== 序列化 ==========
    def serialize(self) -> Dict[str, Any]:
        data = super().serialize()
        data.update({
            "is_DW": self.is_DW,
            "is_csm": self.is_csm,
            "parallelism": self.parallelism,
            "input_size": self.input_size,
            "output_size": self.output_size,
            "is_bypass": self.is_bypass,
        })
        return data

    @classmethod
    def _deserialize_create(cls, data: Dict[str, Any]) -> 'ConvNode':
        return cls(
            name=data["name"],
            view_name=data.get("view_name", data["name"]),
            input_size=data["input_size"],
            output_size=data["output_size"],
            is_csm=data.get("is_csm", False),
            is_DW=data.get("is_DW", False),
        )

    def _deserialize_extra(self, data: Dict[str, Any]):
        self.parallelism = data.get("parallelism", 1)
        if data.get("is_bypass", False):
            self.set_bypass(True)

    @classmethod
    def is_creatable(cls):
        return False

    @classmethod
    def is_deletable(cls):
        return False
