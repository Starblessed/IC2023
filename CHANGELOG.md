
## 0.1.2 (27/12/2023 - 05/01/2024)

- Métricas agora têm o formato de classe (ValidationMetrics), podendo ser instanciada como várias métricas diferentes;
- Data dos dados pluviométricos agora se encontram no formato MM/DD/AAAA para maior compatibilidade com os dados de qualidade da água.

## 0.1.3 (05/01/2024 - 24/01/2024)

- Elaborado protocolo de plotagem de séries temporais de um único ponto de coleta (protocol_ex);
- Elaborada função de interpolação de variáveis para dados faltantes;
- Elaborada função para realizar a união dos dados pluviométricos e de qualidade da água.

## 0.1.4 (24/01/2024 - Pendente)

- Data dos dados pluviométricos agora se encontram no formato AAAA-MM-DD;
- Ajustada a função de união de dados pluviométricos e qualidade da água (merge_pluvio), permitindo utilizar a hora mais próxima;
- Adicionada opção de especificação de horário de referência para os dados pluviométricos na função merge_pluvio();
- Adicionar opção de uso dos dados pluviométricos do dia anterior à coleta.