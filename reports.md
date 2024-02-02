## 09/01/2024
### Relação entre Variáveis
#### Relatório
1. O fator de correlação entre o pH e a DBO (mg/L) é igual a -0.370 (37,0% negativo), indicando relação inversamente proporcional;
2. Foi encontrado um valor semelhante para o fator de correlação entre os Coliformes Termotolerantes (NMP/100mL) e DBO (mg/L): 0.358 (35,8% positivo).
#### Plano de Ação
1. Estudar a relação entre as variáveis com base na literatura
2. Observar relação entre a precipitação, features e label **(Feito)**

## 26/01/2024 - 28/01/2024
### Descontinuidade nos Dados de Monitoramento
#### Relatório
1. Foi percebida uma falha nos dados de monitoramento do INEA, onde, no ano de 2013, as horas de coleta não foram especificadas, dificultando cruzamento com os dados pluviométricos.
#### Plano de Ação **(Concluído)**
1. Experimentar ao menos 3 formas de treinar os modelos utilizando os dados pluviométricos:
    - Ignorando os dados do ano de 2013 **(Feito)**
        - Obs.: Ao utilizar este método, o mse (Mean Squared Error) mínimo obtido foi de aproximadamente 0.67 para o modelo DTR (Decision Tree Regressor). O modelo SVR-Poly (Support Vector Regressor - Polynomial Kernel) apresentou resultados semelhantes, com mse mínimo de aproximadamente 0.78.
    - Utilizando horários do AlertaRio padronizados ao invés da hora mais próxima da coleta **(Feito)**
    - Utilização do último horário do dia anterior **(Feito)**
    - Extra: Combinação do método que descarta os dados do ano de 2013 com os dados pluviométricos das últimas 96h antes da coleta pelo órgão fiscalizador (INEA). **(Feito)**
        - Obs.: Esta abordagem demonstrou mse médias melhores, porém mínimas piores. Assim como a primeira abordagem, os melhores modelos foram: DANN (Deep Artificial Neural Networks), SVR-Poly e DTR. No entanto, é válido observar que o modelo RFR (Random Forest Regression) obteve mse mínima de aproximadamente 0.17, potencialmente por overfitting em uma das iterações. Todavia considerar-se-á o potencial deste modelo em futuras iterações, buscando aprimoramentos.

## 28/01/2024
### Nutriente Limitante - Fósforo
#### Relatório
1. Após análise exploratória com matriz correlação, notou-se que o valor de correlação entre os parâmetros DBO (mg/L) e Fósforo Total (mg/L) é relativamente elevado, apresentando valor de aproximadamente 0.58 (58% positivo). Acredita-se que esta relação implique na papel do fósforo como nutriente limitante para algas e cianobactérias.
#### Plano de Ação
1. Verificar artigos nacionais que tratam da dinâmica de nutrientes e cianobacterias no Sistema Lagunar de Jacarepaguá (SLJP);
2. Utilizar o parâmetro Fósforo Total (mg/L) como variável de predição, buscando aperfeiçoar a precisão e acurácia do modelo, aplicando como principal métrica de validação o *F1-Score*;
3. Realizar análise espaço-temporal da distribuição de Fósforo Total (mg/L) no SLJP.

## 02/02/2024
### Regiões Semelhantes por K-Means Clustering
#### Relatório
1. Após *clusterização* dos dados de todos os pontos de coleta (as 4 lagoas) ao longo de todos os anos (2012-2023), foi possível perceber 4 principais grupos comportamentais para os parâmetros de qualidade da água das lagoas. A principal descoberta é o fato de que o ponto **TJ0303** (Início do Canal da Joatinga - Lagoa da Tijuca) tem semelhança maior com os pontos **MR0363** e **MR0369** (Pontos médio e leste da Lagoa de Marapendi), frente ao ponto irmão **TJ0306** na Lagoa da Tijuca. 4 variáveis foram utilizadas no processo de *clusterização*:
    - pH;
    - Oxigênio Dissolvido (OD (mg/L));
    - Demanda Bioquímica de Oxigênio (DBO (mg/L));
    - Fósforo Total (mg/L).
#### Plano de Ação
1. Utilizar um teste estatístico para comprovar e quantificar a semelhança entre os pontos **TJ0303**, **MR0363** e **MR0369**.
2. Investigar os motivos por trás da semelhança entre os pontos.