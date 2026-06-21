# Gêmeo Digital para Otimização de Irrigação em Culturas de Pequena Escala

> Projeto desenvolvido na disciplina **INF0549 — Gêmeos Digitais: Conceitos, Arquiteturas e Aplicações Práticas**, ministrada pelo Prof. [Iwens Gervasio Sene Junior](http://www.docente.ufg.br/iwens) no Instituto de Informática da Universidade Federal de Goiás (UFG).

Sistema de gêmeo digital que integra leituras de sensores de campo (temperatura e umidade do solo e do ar), dados meteorológicos do INPE e um modelo preditivo baseado na equação de Penman-Monteith (FAO-56) para antecipar quando e quanto irrigar em culturas de pequena escala.

## Visão geral

O projeto resolve três desafios estruturais comuns em agricultura digital de pequena escala: a ausência de hardware instalado para coleta de dados reais, a ausência de rótulos supervisionados para treinamento de modelos preditivos, e a falta de informações detalhadas do sistema solo-planta (análise pedológica, fase fenológica, coeficientes culturais específicos). A solução combina geração de dataset sintético fisicamente coerente, uso da equação de Penman-Monteith como pseudo-rótulo supervisionado, e adoção explícita de premissas pedológicas tabuladas pela FAO.

O sistema atualmente opera em modo de demonstração com dados sintéticos calibrados para o clima de Goiânia. O motor de física rodando no dashboard reproduz, em tempo real no navegador, o ciclo completo de evapotranspiração, decaimento da umidade do solo, eventos estocásticos de chuva e decisões automáticas e manuais de irrigação.

## Demonstração

O dashboard está disponível em produção via Firebase Hosting. A demonstração ao vivo inclui troca dinâmica de imagens conforme o estado do sistema (dia/noite × clima × umidade do solo), controles manuais para irrigação imediata com volume parametrizável, botão de pausa da irrigação automática, e toggle para desligar chuvas durante a demonstração.

## Estrutura do repositório

```
gemeo-digital-irrigacao/
├── README.md                       Este documento
├── .gitignore                      Arquivos não versionados
├── requirements.txt                Dependências Python
├── firebase.json                   Configuração do Firebase Hosting
├── src/                            Código-fonte Python
│   ├── gerar_dataset.py            Gerador de dataset sintético (390 dias)
│   ├── modelo_irrigacao.py         Treinamento do XGBoost MultiOutput
│   ├── coletar_inpe.py             Coletor de dados meteorológicos (CPTEC + Open-Meteo)
│   ├── firebase_writer.py          Publicador de leituras no Firebase Realtime DB
│   └── simulador_ao_vivo.py        Simulador de leitura ao vivo (modo offline)
├── data/                           Datasets gerados
│   ├── dataset_historico.csv       18.720 registros para treinamento
│   └── dataset_futuro.csv          14 dias para simulação ao vivo
├── public/                         Frontend (Firebase Hosting)
│   ├── index.html                  Dashboard completo com motor de física
│   └── imagens/                    18 cenários visuais (dia/noite × clima × solo)
└── docs/                           Documentação técnica
    └── resumo_metodologico.docx    Resumo metodológico da proposta
```

## Componentes principais

### Geração do dataset sintético

O script `src/gerar_dataset.py` produz dois conjuntos de dados estruturalmente idênticos, com granularidade de 30 minutos: um dataset histórico de 390 dias para treinamento e um dataset futuro de 14 dias para simulação ao vivo. As variáveis geradas incluem temperatura e umidade do ar e do solo, radiação solar, velocidade do vento, ETo calculada via Penman-Monteith FAO-56, e eventos de irrigação com volume e confiança. Os parâmetros estão calibrados para o clima do cerrado goiano (latitude -16,7°, altitude 748 m).

### Modelo preditivo

O script `src/modelo_irrigacao.py` treina um XGBoost MultiOutputRegressor que prediz simultaneamente três alvos: horas até a próxima irrigação, volume de água em L/m² e confiança da decisão em percentual. A engenharia de features inclui janelas deslizantes de 6 horas (média, desvio padrão, mínimo, máximo) para todas as variáveis dos sensores, codificação cíclica de hora e dia do ano via funções seno e cosseno, e ETo acumulada nas últimas 24 horas como proxy de déficit hídrico.

A pseudo-label adotada para tarefa supervisionada é a equação de Penman-Monteith FAO-56, que substitui a ausência de rótulos reais por um sinal físico fundamentado. O modelo aprende a antecipar o que a equação física faria, enriquecida por informações de lookahead da previsão meteorológica.

### Coletor de dados externos

O script `src/coletar_inpe.py` integra duas fontes meteorológicas em cascata. A fonte primária é a API XML pública do CPTEC/INPE, que retorna previsão de sete dias por código de cidade. A fonte secundária é a API Open-Meteo, gratuita e alinhada com padrões WMO, que fornece previsões horárias e calcula a ETo via FAO-56 pelo próprio serviço. A comparação entre ETo calculada localmente e ETo estimada pelo INPE é exibida no dashboard como sinal cruzado de validação.

### Integração com Firebase

O script `src/firebase_writer.py` publica leituras no Firebase Realtime Database em formato JSON, com sincronização instantânea para o dashboard via WebSocket. O dashboard (`public/index.html`) lê esses dados em tempo real sem necessidade de polling ou refresh.

### Dashboard

O arquivo `public/index.html` é um aplicativo de página única que apresenta cinco páginas navegáveis (Visão Geral, Sensores, Clima/INPE, Simulação, Alertas), com motor de física integrado que reproduz o ciclo solo-planta-atmosfera no navegador. As imagens de fundo trocam dinamicamente conforme o estado do sistema, e os badges agronômicos classificam as condições atuais (Crítico, Baixo, Adequado, Elevado, Saturado).

## Como executar localmente

### Pré-requisitos

- Python 3.10 ou superior
- Node.js 18 ou superior (apenas para deploy no Firebase Hosting)
- Conta no Firebase com projeto criado e Realtime Database habilitado

### Instalação das dependências Python

```bash
pip install -r requirements.txt
```

### Geração do dataset

```bash
python src/gerar_dataset.py
```

Gera os arquivos `data/dataset_historico.csv` e `data/dataset_futuro.csv`.

### Treinamento do modelo

```bash
python src/modelo_irrigacao.py
```

Produz `modelo_irrigacao.pkl`, `scaler_irrigacao.pkl` e `feature_cols.pkl` na raiz do projeto. Esses arquivos não são versionados (estão no `.gitignore`) e devem ser regenerados localmente.

### Publicação no Firebase

1. Acesse `console.firebase.google.com`, crie um projeto e habilite o Realtime Database em modo de teste.
2. Em Project Settings → Service Accounts, gere uma chave privada e salve o JSON como `serviceAccountKey.json` na raiz do projeto.
3. Edite a variável `DATABASE_URL` em `src/firebase_writer.py` com a URL do seu banco.
4. Execute o publicador:

```bash
python src/firebase_writer.py --rapido 2     # Demo rápida
python src/firebase_writer.py --real         # 1 leitura a cada 30 min (produção)
python src/firebase_writer.py --offline      # Sem conectar ao Firebase
```

### Deploy do dashboard

```bash
firebase login
firebase deploy --only hosting
```

## Metodologia

O documento completo da metodologia adotada está em `docs/resumo_metodologico.docx`. Os tópicos principais são:

- Justificativa para a geração de dataset sintético ao invés do uso de datasets públicos.
- Fontes e equações utilizadas para gerar cada variável do dataset.
- Adoção da equação de Penman-Monteith FAO-56 como pseudo-rótulo supervisionado.
- Estratégia de validação sobre o conjunto sintético com split temporal 70/30.
- Convergência progressiva: como o modelo migra de dados sintéticos para dados reais à medida que sensores forem instalados.

## Referências

- ALLEN, R. G.; PEREIRA, L. S.; RAES, D.; SMITH, M. *Crop evapotranspiration — Guidelines for computing crop water requirements*. FAO Irrigation and Drainage Paper No. 56. Roma: Organização das Nações Unidas para a Alimentação e a Agricultura (FAO), 1998.
- CHEN, T.; GUESTRIN, C. XGBoost: A Scalable Tree Boosting System. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 2016, p. 785–794.
- INPE. CPTEC API de Previsão Numérica de Tempo e Clima. Disponível em http://servicos.cptec.inpe.br/XML/.

## Autores

Trabalho desenvolvido por:

- **Wallisson Policarpo**
- **Giulio Henrique**
- **Lívia Maria**

Bacharelado em Inteligência Artificial — Instituto de Informática, Universidade Federal de Goiás (UFG).

Disciplina **INF0549 — Gêmeos Digitais: Conceitos, Arquiteturas e Aplicações Práticas**, ministrada pelo Prof. [Iwens Gervasio Sene Junior](http://www.docente.ufg.br/iwens).

## Licença

Este projeto está licenciado sob a Licença MIT. Veja o arquivo `LICENSE` para detalhes.
