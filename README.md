Тестовый вариант использования MBart для краткого пересказа новостей.

Если планируете использовать CUDA, подберите совместимую библиотеку, для начала (после установки torch):

1. Проверьте доступность

import torch
print(torch.cuda.is_available())  # Должно вернуть True, если CUDA доступна

2. Устанавливаем через pip необходимую верисю:

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 - для версии 11.8

https://pytorch.org/get-started/locally/ - для справки