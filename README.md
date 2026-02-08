# 🌎 Dashboard de Emissões de Gases de Efeito Estufa

Este projeto entrega um painel interativo para analisar emissões estaduais brasileiras de gases de efeito estufa (GEE). O pipeline combina um notebook exploratório com Pandas/Seaborn e um dashboard Streamlit para destacar tendências por estado, região e período usando dados do SEEG.

## 🚀 Funcionalidades

- Visualização das emissões anuais agregadas por estado e por região.
- Ranking dinâmico dos estados que mais emitem em qualquer intervalo selecionado.
- Mapa interativo com intensidade média das emissões usando PyDeck.
- Correlação entre anos para identificar padrões temporais.
- Exportação em CSV do recorte filtrado diretamente pelo dashboard.

## � Métricas Principais

- Emissões totais (t CO₂e) no período selecionado.
- Variação percentual ano a ano.
- Média de emissões por estado.
- Estado destaque com maior volume de emissões no intervalo.

## 🛠️ Tecnologias Utilizadas

- Python 3.9+
- Pandas & NumPy para preparação e análise de dados.
- Streamlit para o dashboard interativo.
- Plotly Express e PyDeck para visualizações.
- Seaborn/Matplotlib no notebook exploratório.

## � Como Executar

1. **Obtenha os dados brutos**: baixe `dados_gases.xlsx` (aba "GEE Estados") em [seeg.eco.br/download](http://seeg.eco.br/download) e coloque-o na raiz do projeto.
2. **(Opcional) Gere o CSV tratado**: execute todas as células de `gases_poluentes.ipynb`. O notebook cria `emissoes_estado_filtrado.csv`.
3. **Instale as dependências principais**:
   ```bash
   pip install streamlit pandas numpy plotly pydeck seaborn matplotlib openpyxl
   ```
4. **Execute o dashboard**:
   ```bash
   streamlit run app_dashboard.py
   ```
5. Acesse o endereço exibido no terminal (normalmente http://localhost:8501) e utilize os filtros laterais para personalizar a análise.

## 📋 Requisitos

- Python 3.9 ou superior.
- Bibliotecas listadas acima (ou configure um `requirements.txt` com as mesmas dependências).
- Arquivo `dados_gases.xlsx` disponível localmente para reconstruir a base quando necessário.

## 📊 Estrutura do Projeto

- `app_dashboard.py` – implementação do dashboard com KPIs, gráficos Plotly e mapa PyDeck.
- `gases_poluentes.ipynb` – notebook com storytelling completo, limpeza da base e exportação do CSV final.
- `dados_gases.xlsx` – arquivo bruto do SEEG (não incluso no repositório por tamanho/licença).
- `emissoes_estado_filtrado.csv` – dataset derivado pronto para alimentar o dashboard (gerado automaticamente se ausente).

## � Visualizações

1. **Visão Geral** com KPIs principais.
2. Linha temporal das emissões agregadas (Plotly).
3. Pizza da participação por região.
4. Barra com top emissores (top 10 estados).
5. Mapa PyDeck destacando intensidade média anual.
6. Matriz de correlação entre anos e tabela estatística detalhada por estado.

## 📝 Licença

Este projeto é distribuído sob a licença MIT.

---
Projeto desenvolvido por **David** para portfólio de dados ambientais. Feedbacks e sugestões são bem-vindos!
