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
2. Experimentar ao menos 3 formas de treinar os modelos utilizando os dados pluviométricos:
    - Ignorando os dados do ano de 2013 **(Feito)**
        - Obs.: Ao utilizar este método, o mse (Mean Squared Error) mínimo obtido foi de aproximadamente 0.67 para o modelo DTR (Decision Tree Regressor). O modelo SVR-Poly (Support Vector Regressor - Polynomial Kernel) apresentou resultados semelhantes, com mse mínimo de aproximadamente 0.78.
    - Utilizando horários do AlertaRio padronizados ao invés da hora mais próxima da coleta **(Feito)**
    - Utilização do último horário do dia anterior **(Feito)**
    - Extra: Combinação do método que descarta os dados do ano de 2013 com os dados pluviométricos das últimas 96h antes da coleta pelo órgão fiscalizador (INEA). **(Feito)**
        - Obs.: Esta abordagem demonstrou mse médias melhores, porém mínimas piores. Assim como a primeira abordagem, os melhores modelos foram: DANN (Deep Artificial Neural Networks), SVR-Poly e DTR. No entanto, é válido observar que o modelo RFR (Random Forest Regresson) obteve mse mínima de aproximadamente 0.17, potencialmente por overfitting em uma das iterações. Todavia considerar-se-á o potencial deste modelo em futuras iterações, buscando aprimoramentos.