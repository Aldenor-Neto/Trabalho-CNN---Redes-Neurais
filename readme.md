# IFCE Campus Maracanaú

**Disciplina:** Redes Neurais Artificiais (RNA)  
**Professor:** Amauri Holanda de Souza  
**Aluno:** Francisco Aldenor Silva Neto  
**Matrícula:** 20221045050117

---

# Relatório de Experimento com Redes Neurais Convolucionais

## Introdução

Este relatório tem como objetivo apresentar os resultados de uma série de experimentos realizados com Redes Neurais Convolucionais (CNNs) aplicadas aos datasets **MNIST** e **CIFAR-10**. O principal objetivo foi comparar o desempenho de diferentes configurações de CNN, variando o número de camadas convolucionais e filtros, tanto em termos de **perda** quanto em **acurácia** nos datasets mencionados.

## Metodologia

### Redes Neurais Convolucionais

Para este experimento, utilizou-se uma arquitetura básica de CNN. As variações incluídas foram:

1. **Número de Camadas Convolucionais**: As CNNs foram configuradas com 2 ou 3 camadas convolucionais.
2. **Número de Filtros**: Utilizou-se CNNs com 32 e 64 filtros em cada camada convolucional.

O modelo foi treinado utilizando a função de perda **Cross Entropy Loss** e o otimizador **Adam**, com uma taxa de aprendizado de 0.001. Os experimentos foram realizados por 10 épocas para cada configuração de modelo.

### Datasets

Dois datasets foram utilizados nos experimentos:
- **MNIST**: Um dataset de dígitos manuscritos em escala de cinza com imagens de tamanho 28x28 e uma única camada de canal de entrada.
- **CIFAR-10**: Um dataset com imagens coloridas de 32x32 distribuídas em 10 classes, com 3 canais de entrada (RGB).

### Configurações dos Experimentos

As seguintes configurações foram testadas:

1. **Configuração 1 - MNIST**
   - Camadas convolucionais: 2
   - Filtros: 32
   - Tamanho da imagem: 28x28
   - Canais de entrada: 1

2. **Configuração 2 - MNIST**
   - Camadas convolucionais: 3
   - Filtros: 64
   - Tamanho da imagem: 28x28
   - Canais de entrada: 1

3. **Configuração 3 - CIFAR-10**
   - Camadas convolucionais: 2
   - Filtros: 32
   - Tamanho da imagem: 32x32
   - Canais de entrada: 3

4. **Configuração 4 - CIFAR-10**
   - Camadas convolucionais: 3
   - Filtros: 64
   - Tamanho da imagem: 32x32
   - Canais de entrada: 3

## Resultados

Os resultados foram obtidos após o treinamento de 10 épocas para cada configuração. Os gráficos a seguir mostram a curva de perda (**Loss Curve**) para cada configuração, permitindo a comparação visual da convergência dos modelos ao longo das épocas.

### Configuração 1: MNIST com 2 camadas convolucionais e 32 filtros

- **Perda ao final da época 10**: 0.0114
- **Acurácia**: 99.03%

![Curva de Perda - MNIST com 32 Filtros](Imagens/Loss%20Curve%20-%20MNIST%20-%20Filters%2032.jpg)

#### Resultados por época:
```
Época 1: Perda: 0.18385019775198053
Época 2: Perda: 0.060817406763874774
Época 3: Perda: 0.04237998684973462
Época 4: Perda: 0.03360821752620837
Época 5: Perda: 0.0268727101288229
Época 6: Perda: 0.02274255602367854
Época 7: Perda: 0.02003783580401312
Época 8: Perda: 0.01617412261975236
Época 9: Perda: 0.015333169517076901
Época 10: Perda: 0.011384926420859066
```

### Configuração 2: MNIST com 3 camadas convolucionais e 64 filtros

- **Perda ao final da época 10**: 0.0087
- **Acurácia**: 99.24%

![Curva de Perda - MNIST com 64 Filtros](Imagens/Loss%20Curve%20-%20MNIST%20-%20Filters%2064.jpg)

#### Resultados por época:
```
Época 1: Perda: 0.16302646061768933
Época 2: Perda: 0.04359750169987626
Época 3: Perda: 0.03021662253295697
Época 4: Perda: 0.025957261116170922
Época 5: Perda: 0.019692639852205027
Época 6: Perda: 0.016182063290179474
Época 7: Perda: 0.015216261316234169
Época 8: Perda: 0.012127258742448674
Época 9: Perda: 0.011224669979324382
Época 10: Perda: 0.008749728613191794
```

### Configuração 3: CIFAR-10 com 2 camadas convolucionais e 32 filtros

- **Perda ao final da época 10**: 0.7095
- **Acurácia**: 69.88%

![Curva de Perda - CIFAR-10 com 32 Filtros](Imagens/Loss%20Curve%20-%20CIFAR10%20-%20Filters%2032.jpg)

#### Resultados por época:
```
Época 1: Perda: 1.4181570801741021
Época 2: Perda: 1.0829536960557904
Época 3: Perda: 0.9626900797990887
Época 4: Perda: 0.8980992000045069
Época 5: Perda: 0.847045872522437
Época 6: Perda: 0.8098347007542315
Época 7: Perda: 0.7793136276781102
Época 8: Perda: 0.7537860756411272
Época 9: Perda: 0.7287581711245315
Época 10: Perda: 0.7095023627628756
```

### Configuração 4: CIFAR-10 com 3 camadas convolucionais e 64 filtros

- **Perda ao final da época 10**: 0.4279
- **Acurácia**: 75.04%

![Curva de Perda - CIFAR-10 com 64 Filtros](Imagens/Loss%20Curve%20-%20CIFAR10%20-%20Filters%2064.jpg)

#### Resultados por época:
```
Época 1: Perda: 1.3795875501449761
Época 2: Perda: 0.9711265486219655
Época 3: Perda: 0.8173393317119545
Época 4: Perda: 0.7201290439690471
Época 5: Perda: 0.6524606653491555
Época 6: Perda: 0.5921993656917606
Época 7: Perda: 0.5457428269221655
Época 8: Perda: 0.5032934140595023
Época 9: Perda: 0.45982771900380054
Época 10: Perda: 0.4279571445206242
```

### Comparação Geral das Curvas de Perda

Para uma melhor comparação entre as diferentes configurações, foi gerado um gráfico combinando as curvas de perda para todas as configurações testadas.

![Curva de Perda Combinada](Imagens/Combined%20Loss%20Curves.jpg)

## Discussão

Os resultados demonstram um desempenho excelente no dataset **MNIST**, com acurácias superiores a 99% em ambas as configurações testadas. No entanto, o modelo com 3 camadas convolucionais e 64 filtros teve uma leve vantagem, com uma acurácia de **99.24%**, enquanto o modelo com 2 camadas alcançou **99.03%**. Isso sugere que o aumento na complexidade da rede trouxe melhorias, ainda que marginais, para o reconhecimento de dígitos manuscritos.

No caso do dataset **CIFAR-10**, observamos que o desempenho é substancialmente inferior quando comparado ao MNIST, o que pode ser explicado pela maior complexidade das imagens coloridas em CIFAR-10. No entanto, novamente, o modelo com 3 camadas convolucionais e 64 filtros apresentou melhor desempenho (**73.52%** de acurácia), em comparação ao modelo com 2 camadas e 32 filtros (**69.88%**). A perda também foi menor no modelo mais complexo.

Esses resultados indicam que redes mais profundas e com mais filtros tendem a oferecer melhor desempenho em problemas de classificação de imagens mais complexas, como o CIFAR-10, mas o impacto em datasets mais simples como o MNIST é menos significativo.

## Conclusão

Este experimento comparou o desempenho de diferentes arquiteturas de Redes Neurais Convolucionais em dois datasets populares: MNIST e CIFAR-10. Os resultados confirmam que, para problemas mais complexos como o CIFAR-10, o aumento da profundidade da rede e do número de filtros melhora a acurácia. Já para datasets mais simples como o MNIST, essas modificações trazem melhorias mais discretas.

## Referências

Repositório do projeto no GitHub: [GitHub Repository](https://github.com/Aldenor-Neto/Trabalho-CNN---Redes-Neurais)
