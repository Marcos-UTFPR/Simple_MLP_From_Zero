import os
import sys
import time
import random
from NeuralNetwork import *

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Constantes -----------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Só vão ser alteradas dentro do desenvolvimento

# Cores
PURPLE = '\033[95m'
CYAN = '\033[96m'
DARKCYAN = '\033[36m'
BLUE = '\033[94m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLACK = '\x1b[30m'
BRIGHTBLACK   = '\033[30m'
WHITE   = '\033[37m'
MAGENTA = '\033[35m'
RED = '\033[91m'
# Estilos de texto
BLINK = "\033[5m"
BOLD = '\033[1m'
UNDERLINE = '\033[4m'
END = '\033[0m' # Reseta toda a formatação
# Vetor completo
COLORS = [PURPLE, CYAN, DARKCYAN, BLUE, GREEN, YELLOW, RED, MAGENTA, BLACK, BRIGHTBLACK, WHITE, BLINK, BOLD, UNDERLINE, END]

FRUIT_AMOUNT = 1 # Quantidade de frutas inicialmente (Default = 1)

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Variáveis Globais ----------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------

lines_number = 10 # Número de linhas do tabuleiro - Default: 8
columns_number = 25 # Número de colunas do tabuleiro (cells de cada linha) - Default: 15

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Exceções -------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------

class DefeatException(Exception):
    def __init__(self, message=("Colisão! Você perdeu...")):
        # Call the base class constructor with the parameters it needs
        super(DefeatException, self).__init__(message)

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Funções auxiliares ---------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------

def typewriterPrint(message): # Um print lento com efeito de "máquina de escrever"
    for x in message:
        print(x, end='')
        sys.stdout.flush()
        time.sleep(0.1)
    print('\n', end='') # Pulando linha

def clear(): # Limpa o terminal
    os.system('cls' if os.name == 'nt' else 'clear')

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Outros ---------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------

def doNothing():
    pass

def doNothingForApproximately(seconds): # Um time.sleep() bem piorado
    # 1 ciclo quase é 1 segundo exato
    for i in range(seconds):
        i = 0
        while i < 16706910:
            pass
            i += 1

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Classes principais ---------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------

class base_element:
    # Atributos: name, current_cell
    def __init__(self, name, cell=None):
        self.name = name # Nome do tipo de peça dela
        self.current_cell = cell # Casa onde ela está atualmentes

    # ---------------------------------------

    def print(self):
        pass

    # ---------------------------------------

    def set_cell(self, new_cell):
        self.current_cell = new_cell

    # ---------------------------------------

# ----------------------------------------------------------------------------------------------------

class empty(base_element):
    # Atributos: name, current_cell
    def __init__(self, cell=None):
        super().__init__("empty", cell)

    # ---------------------------------------

    def print(self):
        return " "

    # ---------------------------------------

# ----------------------------------------------------------------------------------------------------

class world_wall(base_element): # Unbreakable
    # Atributos: name, current_cell
    def __init__(self, cell=None):
        super().__init__("wall", cell)

    # ---------------------------------------

    def print(self):
        return str(YELLOW+"█"+END)

    # ---------------------------------------

# ----------------------------------------------------------------------------------------------------

class fruit(base_element): # Unbreakable
    # Atributos: name, current_cell
    def __init__(self, cell=None):
        super().__init__("fruit", cell)

    # ---------------------------------------

    def print(self):
        return str(RED+"█"+END)

    # ---------------------------------------

# ----------------------------------------------------------------------------------------------------

class cell:
    def __init__(self, number, own_line):
        self.cellNumber = number
        self.own_line = own_line # own_line: Linha em que a célula está
        self.current_element = empty(cell=self)

    # ---------------------------------------

    def print(self):
        #return " "+self.current_element.print()+" "
        return self.current_element.print()

    # ---------------------------------------

    def alter_element(self, new_element):
        del self.current_element
        self.current_element = new_element
        if self.current_element.current_cell != self:
            self.current_element.set_cell(self)

    # ---------------------------------------

# ----------------------------------------------------------------------------------------------------

class line:
    # Atributos: cells e lineNumber (número da linha)
    def __init__(self, number):
        number = int(number)
        if number > 0 and number < lines_number+1: # De 1 a lines_number
            self.lineNumber = number
            self.cells = {}
            for i in range(1,columns_number+1): # Quinze células, 1 a 15
                self.cells[str(i)] = cell(i, self)

    # ---------------------------------------

    def print(self):
        #OBS: Os prints das células retornam strings
        stringPrint = ""
        for cell in self.cells.values():
            stringPrint += (f"{cell.print()}") # Cada célula printa o conteúdo do seu elemento
        return stringPrint

    # ---------------------------------------

# ----------------------------------------------------------------------------------------------------

class table:
    # Atributos: lines
    def __init__(self):
        self.lines = {} # Sim, um dicionário, não lista
        for i in range(1,lines_number+1): # Oito linhas, 1 a 8 por padrão
            #print(str(i))
            self.lines[str(i)] = line(i)

    # ---------------------------------------

    def print(self):
        #OBS: Os prints das linhas retornam strings
        print('-----------'+('-'*(3*columns_number))+'----------------'+'\n\t   ',end="")
        for i in range(1,columns_number+1):
            print((" "+str(i)+(" " if i < 9 else "")),end="")
        print("")
        for line in self.lines.values():
            print(f"\t{line.lineNumber}."+(" " if line.lineNumber <= 9 else "")+f"{line.print()}\n",end="")
        print('-----------'+('-'*(3*columns_number))+'----------------'+'\n')

    # ---------------------------------------

    def starting_elements(self,mainPlayer):
        # Colocando os World Walls ao redor da table
        for cell in self.lines["1"].cells.values():
            cell.alter_element(world_wall("world_wall"))
        for cell in self.lines[str(lines_number)].cells.values():
            cell.alter_element(world_wall("world_wall"))
        
        for line in self.lines.values():
            line.cells["1"].alter_element(world_wall("world_wall"))
            line.cells[str(columns_number)].alter_element(world_wall("world_wall"))

        self.random_fruit(number=FRUIT_AMOUNT)

        # Colocando o jogador
        while True:
            line_number = str(random.randint(2,lines_number-1))
            cell_number = str(random.randint(2,columns_number-1))
            if (self.lines[line_number].cells[cell_number].current_element.name) == "empty":
                self.lines[line_number].cells[cell_number].alter_element(mainPlayer)
                break
        
    # ---------------------------------------

    def random_fruit(self, number=1):
        # Colocando uma fruta aleatória
        for i in range(0,number):
            while True: # Tenta spawna até achar uma célula vazia
                line_number = str(random.randint(2,lines_number-1))
                cell_number = str(random.randint(2,columns_number-1))
                if (self.lines[line_number].cells[cell_number].current_element.name) == "empty":
                    self.lines[line_number].cells[cell_number].alter_element(fruit())
                    break

    # ---------------------------------------

    def get_position(self, column_number, line_number):
        #print(self.lines[str(line_number)].cells.keys())
        return self.lines[str(line_number)].cells[str(column_number)]

    # ---------------------------------------

    def get_fruit_position(self): # Retorna a posição da primeira fruta
        #print(self.lines[str(line_number)].cells.keys())
        for line in self.lines.values():
            for cell in line.cells.values():
                if cell.current_element.name == "fruit":
                    return cell.cellNumber, line.lineNumber

    # ---------------------------------------

# ----------------------------------------------------------------------------------------------------

class player(base_element): # Head
    #Atributos: current_cell, next_segment, previous_cell, score
    def __init__(self, color, name):
        assert color in COLORS
        self.color = color
        self.next_segment = None # Próximo body (inicia só com a Head) - Funciona como uma lista encadeada
        self.previous_cell = None
        self.score = 0 # Aumenta 1 a cada fruta consumida
        super().__init__(name, cell)

    # ---------------------------------------

    def print(self):
        return self.color+'☺'+ END

    # ---------------------------------------

    def move(self, new_cell):
        self.previous_cell = self.current_cell
        super().set_cell(new_cell)
        self.current_cell.alter_element(self)
        self.move_body()

    # ---------------------------------------

    def move_body(self):
        if self.next_segment is not None:
            self.next_segment.move(self.previous_cell)
        else:
            self.previous_cell.alter_element(empty())
        
    # ---------------------------------------

    def consume(self, new_cell):
        self.move(new_cell)
        self.score += 1
        if self.next_segment is not None:
            self.next_segment.grow()
        else:
            self.next_segment = body(self, cell=self.previous_cell)

    # ---------------------------------------

    def get_current_position(self):
        return int(self.current_cell.cellNumber), int(self.current_cell.own_line.lineNumber)

    # ---------------------------------------

# ----------------------------------------------------------------------------------------------------

class body(base_element): # Corpo
    #Atributos: current_cell
    def __init__(self, player, cell=None):
        self.player_head = player
        self.color = self.player_head.color
        self.next_segment = None # Próximo body (inicia com nada)
        self.previous_cell = None
        super().__init__("body", cell)

    # ---------------------------------------

    def print(self):
        return self.color+'O'+ END

    # ---------------------------------------

    def move(self, new_cell):
        self.previous_cell = self.current_cell
        super().set_cell(new_cell)
        self.current_cell.alter_element(self)
        self.move_body()

    # ---------------------------------------

    def move_body(self):
        if self.next_segment is not None:
            self.next_segment.move(self.previous_cell)
        else:
            self.previous_cell.alter_element(empty())

    # ---------------------------------------

    def grow(self):
        if self.next_segment is None: # Se chegou no final
            self.next_segment = body(self.player_head, cell=self.previous_cell)
        else:
            self.next_segment.grow()

    # ---------------------------------------

    def get_current_position(self):
        return int(self.current_cell.cellNumber), int(self.current_cell.own_line.lineNumber)

    # ---------------------------------------



# ----------------------------------------------------------------------------------------------------

class game:
    # Atributos: mainTable e mainPlayer
    def __init__(self, mainPlayer):
        self.mainPlayer = mainPlayer
        self.mainTable = table()

    # ---------------------------------------

    def print(self): # Print usada para chamar o método que gera a tabela
        clear()
        print('-----------'+('-'*(3*columns_number))+'----------------'+'\n')
        print(f" Score:{self.mainPlayer.score}")
        fruit_position = self.mainTable.get_fruit_position()
        if fruit_position is None:
            # Se a Fruta simplesmente não aparecer por razões misteriosas
            self.mainTable.random_fruit() # Spawna uma fruta de forma forçada
            fruit_position = self.mainTable.get_fruit_position()
        print(f"Posição da fruta: Coluna {fruit_position[0]} | Linha {fruit_position[1]}")
        self.mainTable.print()
        print('-----------'+('-'*(3*columns_number))+'----------------')
    
    # ---------------------------------------

    def start(self):
        try:
            self.mainTable.starting_elements(self.mainPlayer)
            while True:
                time.sleep(0.2)
                clear()
                self.round()
        except DefeatException as e:
            self.ending(e)

    # ---------------------------------------

    def round(self):
        self.print()
        self.move()

    # ---------------------------------------

    def move(self):
        possible_directions = ["up", "down", "left", "right", "cima", "baixo", "esquerda", "direita"]
        #direction = input("Por favor, informe uma direção: ").lower()  # Escolha do jogador
        #direction = possible_directions[random.randint(0,len(possible_directions)-1)]   # Movimento aleatório
        direction = self.fruit_direction(possible_directions) # Caminho direto pra fruta
        #direction = self.perfect_path(possible_directions) # Algoritmo para zerar o jogo
        while True:
            if direction not in possible_directions:
                direction = input("Por favor, informe uma direção válida [up, down, left, right, cima, baixo, esquerda ou direita]: ").lower()
            else:
                break
        current_player_column_number, current_player_line_number = self.mainPlayer.get_current_position()

        # Pegando a célula destino
        if direction in ["up", "cima"]:
            new_cell = self.mainTable.get_position(current_player_column_number,current_player_line_number-1)
        elif direction in ["down", "baixo"]:
            new_cell = self.mainTable.get_position(current_player_column_number,current_player_line_number+1)
        elif direction in ["left", "esquerda"]:
            new_cell = self.mainTable.get_position(current_player_column_number-1,current_player_line_number)
        elif direction in ["right", "direita"]:
            new_cell = self.mainTable.get_position(current_player_column_number+1,current_player_line_number)

        if new_cell.current_element.name == "empty":
            self.mainPlayer.move(new_cell)
        elif new_cell.current_element.name == "fruit":
            self.mainPlayer.consume(new_cell)
            self.mainTable.random_fruit()
        elif new_cell.current_element.name == "body":
            raise DefeatException
        else:
            print(RED+"Erro! Casa ocupada!!! Escolha outra!"+END)

        # WIP !!!!!!!!!!!!

    # ---------------------------------------

    def fruit_direction(self, possible_directions):
        current_position = self.mainPlayer.get_current_position()
        fruit_position = self.mainTable.get_fruit_position()
        # [0] é a coluna, [1] é a linha
        if current_position[0] > fruit_position[0]:
            return "left"
        elif current_position[0] < fruit_position[0]:
            return "right"
        elif current_position[1] > fruit_position[1]:
            return "up"
        elif current_position[1] < fruit_position[1]:
            return "down"
        else:
            time.sleep(1)
            return possible_directions[random.randint(0,len(possible_directions)-1)] # Aleatória
            # WIP !!!!!!!!!

    # ---------------------------------------

    def perfect_path(self, possible_directions):
        current_position = self.mainPlayer.get_current_position()
        # WIP !!!!!!!!!!! 

    # ---------------------------------------

    def ending(self, e):
        print("\t"+RED+BOLD+str(e)+END)

    # ---------------------------------------

    def get_estado(self):
        col_cabeca, lin_cabeca = self.mainPlayer.get_current_position()
        fruit_position = self.mainTable.get_fruit_position()
        col_fruta, lin_fruta = fruit_position

        # Checando o que há em cada direção adjacente à cabeça
        def e_perigoso(col, lin):
            try:
                elemento = self.mainTable.get_position(col, lin).current_element.name
                return 1.0 if elemento in ("wall", "body") else 0.0
            except (KeyError, Exception):
                return 1.0  # Fora do tabuleiro = perigo

        bloqueado_cima    = e_perigoso(col_cabeca, lin_cabeca - 1)
        bloqueado_baixo   = e_perigoso(col_cabeca, lin_cabeca + 1)
        bloqueado_esq     = e_perigoso(col_cabeca - 1, lin_cabeca)
        bloqueado_dir     = e_perigoso(col_cabeca + 1, lin_cabeca)

        # Normalizando posições para o intervalo [0, 1]
        # A sigmoid satura com valores grandes — sem isso a rede não aprende
        col_cabeca_n = col_cabeca / columns_number
        lin_cabeca_n = lin_cabeca / lines_number
        col_fruta_n  = col_fruta  / columns_number
        lin_fruta_n  = lin_fruta  / lines_number

        return [
            col_cabeca_n, lin_cabeca_n,
            col_fruta_n,  lin_fruta_n,
            bloqueado_cima, bloqueado_baixo, bloqueado_esq, bloqueado_dir
        ]

    # ---------------------------------------

    def executar_acao(self, acao):
        # 0=cima, 1=baixo, 2=esquerda, 3=direita
        direcoes = ["up", "down", "left", "right"]
        direction = direcoes[acao]

        col_cabeca, lin_cabeca = self.mainPlayer.get_current_position()

        if direction == "up":
            new_cell = self.mainTable.get_position(col_cabeca, lin_cabeca - 1)
        elif direction == "down":
            new_cell = self.mainTable.get_position(col_cabeca, lin_cabeca + 1)
        elif direction == "left":
            new_cell = self.mainTable.get_position(col_cabeca - 1, lin_cabeca)
        elif direction == "right":
            new_cell = self.mainTable.get_position(col_cabeca + 1, lin_cabeca)

        elemento = new_cell.current_element.name

        # Calculando distância antes do movimento para a recompensa de proximidade
        fruit_position = self.mainTable.get_fruit_position()
        dist_antes = abs(col_cabeca - fruit_position[0]) + abs(lin_cabeca - fruit_position[1])

        if elemento == "empty":
            self.mainPlayer.move(new_cell)
            nova_col, nova_lin = self.mainPlayer.get_current_position()
            fruit_position = self.mainTable.get_fruit_position()

            if fruit_position is None:
                self.mainTable.random_fruit()
                fruit_position = self.mainTable.get_fruit_position()

            dist_depois = abs(nova_col - fruit_position[0]) + abs(nova_lin - fruit_position[1])
            recompensa = 1.0 if dist_depois < dist_antes else -1.0
            return recompensa, False

        elif elemento == "fruit":
            self.mainPlayer.consume(new_cell)
            self.mainTable.random_fruit()
            return 10.0, False

        else:  # wall ou body
            raise DefeatException

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Seção principal do código --------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------

def main(): # Função principal
    clear() # Limpa o terminal
    
    # ---------------------------------------
    global gameMain
    the_player = player(BLUE, "player")
    gameMain = game(the_player) # Iniciando com as peças brancas
    gameMain.start()
    # ---------------------------------------

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(YELLOW+"Programa encerrado via terminal..."+END)

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Fim do código --------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------    
