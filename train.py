import time
import sys
from Snake import game, player, BLUE, DefeatException
from NeuralNetwork import Custom_MLP, Layer, DQN_Agent

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Configurações --------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------

EPISODIOS        = 1000   # Total de episódios de treino (Padrão: 1000)
PASSOS_MAX       = 500    # Passos máximos por episódio (evita loop infinito - Padrão: 500)
VISUALIZAR_A_CADA = 50    # Exibe a IA jogando a cada N episódios (Padrão: 50)
DELAY_VISUALIZACAO = 0.15 # Segundos entre cada frame ao visualizar (Padrão: 0.15)

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Rede e Agente --------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------

rede = Custom_MLP([
    Layer(16, 8),
    Layer(16, 16),
    Layer(4, 16)
])
agente = DQN_Agent(rede)

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Funções auxiliares ---------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------

def novo_jogo():
    the_player = player(BLUE, "player")
    snake = game(the_player)
    snake.mainTable.starting_elements(the_player)
    return snake

def rodar_episodio(snake, visualizar=False):
    estado = snake.get_estado()
    passos = 0

    while True:
        if visualizar:
            snake.print()
            print(f" Epsilon: {agente.epsilon:.3f}")
            time.sleep(DELAY_VISUALIZACAO)

        try:
            acao = agente.escolher_acao(estado)
            recompensa, morreu = snake.executar_acao(acao)

            proximo_estado = snake.get_estado()
            agente.memorizar(estado, acao, recompensa, proximo_estado, morreu)
            agente.treinar_passo_unico(estado, acao, recompensa, proximo_estado, morreu)
            agente.treinar_replay()

            estado = proximo_estado
            passos += 1

            if passos >= PASSOS_MAX:
                print("Limite de passos!")
                break

        except DefeatException:
            if visualizar:
                snake.print()
                print(" A cobra morreu!")
                time.sleep(1)
            agente.memorizar(estado, acao, -10.0, estado, True)
            agente.treinar_passo_unico(estado, acao, -10.0, estado, True)
            agente.treinar_replay()
            break

    return snake.mainPlayer.score

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Loop principal -------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------

def main():
    print("Iniciando treino...\n")
    melhor_score = 0

    for episodio in range(1, EPISODIOS + 1):
        visualizar = (episodio % VISUALIZAR_A_CADA == 0)
        snake = novo_jogo()
        score = rodar_episodio(snake, visualizar=visualizar)

        if score > melhor_score:
            melhor_score = score

        agente.decair_epsilon()

        # Log a cada 10 episódios
        if episodio % 1 == 0:
            print(f"Episódio {episodio:>5} / {EPISODIOS} "
                  f"| Score: {score:>3} "
                  f"| Melhor: {melhor_score:>3} "
                  f"| Epsilon: {agente.epsilon:.3f} "
                  f"| Memória: {len(agente.memory):>5}")

    print("\nTreino concluído!")
    print(f"Melhor score: {melhor_score}")

    # Sessão de visualização final — IA jogando sem exploração aleatória
    print("\nSessão final — IA jogando sem aleatoriedade (epsilon = 0)...")
    agente.epsilon = 0.0
    input("Pressione Enter para começar...")
    for partida in range(3):
        snake = novo_jogo()
        score = rodar_episodio(snake, visualizar=True)
        print(f"Partida {partida + 1} — Score final: {score}")
        time.sleep(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTreino interrompido.")
        sys.exit(0)