import matplotlib.pyplot as plt
import pandas as pd

# Selecionar um arquivo para plotar as perdas
path = "losses_sgd.csv"

# Carregar o arquivo CSV
df = pd.read_csv(path)

# Plotar os valores das perdas para o gerador, discriminador e validação
plt.figure(figsize=(10, 6))

# Plotar a perda do gerador
plt.plot(df['Epoch'], df['G Loss'], label='G Loss', color='b')

# Plotar a perda do discriminador
plt.plot(df['Epoch'], df['D Loss'], label='D Loss', color='r')

# Plotar a perda de validação
plt.plot(df['Epoch'], df['Validation Loss'], label='Validation Loss', color='g')

# Definir o limite do eixo x
plt.xlim([0, 105])

# Adicionar rótulos e título
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Using SGD Optimizer for discriminator and images with better resolution')
plt.legend()

# Salvar o gráfico em um arquivo
plt.savefig('plot/sgd_model.png')

# Exibir o gráfico
plt.show()