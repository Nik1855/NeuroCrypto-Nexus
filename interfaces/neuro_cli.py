import cmd
from pyfiglet import Figlet
from modules.self_healing.error_autofix import CodeSurgeon


class NeuroCLI(cmd.Cmd):
    prompt = "NeuroCortex> "
    rooms = {
        "main": "Главный центр управления",
        "ai_lab": "Лаборатория ИИ моделей",
        "trading_floor": "Торговый терминал",
        "sys_clinic": "Системная клиника",
        "hardware_core": "Аппаратное ядро"
    }

    def __init__(self):
        super().__init__()
        self.current_room = "main"
        self.display_welcome()

    def display_welcome(self):
        f = Figlet(font="cybermedium")
        print(f.renderText("NeuroCrypto Nexus"))
        print("Адаптивная нейроморфная система анализа крипторынка")
        print("Введите 'help' для списка команд, 'room' для смены помещения\n")

    def do_room(self, arg):
        """Смена виртуального помещения: room [название]"""
        if arg in self.rooms:
            self.current_room = arg
            print(f"Переход в: {self.rooms[arg]}")
        else:
            print("Доступные помещения:", ", ".join(self.rooms.keys()))

    def do_scan(self, arg):
        """Запуск сканирования рынка: scan [pair] [depth]"""
        from modules.market_analyzers.deep_scanner import run_deep_scan
        run_deep_scan(arg if arg else "all", depth=3)

    def do_heal(self, arg):
        """Авто-лечение системы: heal [error_code]"""
        CodeSurgeon.perform_surgery(error_code=arg)

    def do_quantize(self, arg):
        """Квантование моделей: quantize [model] [precision]"""
        from modules.hardware_accelerators.quantization import dynamic_quantize
        dynamic_quantize(model_name=arg)

    def do_trade(self, arg):
        """Управление торговлей: trade [start|stop|status]"""
        from modules.trading_bots.autonomous_trader import manage_trading
        manage_trading(command=arg)

    def do_exit(self, arg):
        """Выход из системы"""
        print("Деактивация нейроморфного ядра...")
        return True


if __name__ == "__main__":
    NeuroCLI().cmdloop()