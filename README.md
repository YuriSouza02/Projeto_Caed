------------------------------------------------------------------------
PRÉ-REQUISITOS
------------------------------------------------------------------------
* Python 3.8 ou superior instalado.
* Linux (Debian/Ubuntu): Certifique-se de ter o pacote 'python3-venv'
  instalado no sistema antes de rodar pela primeira vez:

  sudo apt update && sudo apt install python3-venv

Considere que todo o desenvolvimento foi feito em ambiente windows. Pode haver passos a mais para tomar caso queira rodar a aplicação em ambiente linux.

------------------------------------------------------------------------
COMO EXECUTAR
------------------------------------------------------------------------
Abra o terminal na pasta raiz do projeto e utilize os comandos abaixo.

[A] Istalação do corpus textual
essa tarefa precisa ser feita de forma manual. dentro da pasta "DataBase" há as 4 sub divisões do corpus textual (Ensino fundamental 1, Ensino fundamental 2, Ensino medio e Ensino superior). Deverá ser inserido nas suas respectivas pastas os textos de cada nivel textual os textos correspondentes. Os textos deveram esta separados em arquivos diferentes e todos em TXT.

[B] PRIMEIRA VEZ RODANDO O PROJETO
O script dependencias.py cuida de todas as dependencias do projeto. O tempo de download e intalação gira entorno de 10 minutos.

Windows:
  python dependencias.py

Linux / macOS:
  python3 dependencias.py


[C] TREINAR O MODELO
treine o algoritmo de predição utilizando:

Windows:
  .venv\Scripts\python Treinamento.py

Linux / macOS:
  .venv/bin/python Treinamento.py


[D] EXECUTAR A APLICAÇÂO WEB
Como a .venv já estará configurada:

  Execute diretamente o servidor:

  Windows:
    python Iniciar.py

  Linux / macOS:
    python Iniciar.py