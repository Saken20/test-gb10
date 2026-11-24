# Бенчмарк голосового пайплайна

[English version](../README.md)

Показывает, насколько быстро, для скольких пользователей и насколько качественно цепочка речь + LLM (**ASR → NLU →
LLM → TTS** или любая её часть) работает на вашем железе. Для тех, кто выбирает машину, модель или настройки сервера
под голосового ассистента или чат-бота. Работает с любым OpenAI-совместимым сервером (vLLM, SGLang, llama.cpp, TGI
и другими), простыми HTTP/JSON-сервисами и плагинами.

## Быстрый старт: замер LLM в 3 шага

Нужен запущенный OpenAI-совместимый сервер LLM, например `vllm serve Qwen/Qwen3-8B --port 8000`
(скрипты — в [`setup/`](../setup/README.md)). Нет GPU под рукой? Попробуйте [пробный прогон](#пробный-прогон-без-gpu).
Команды — для bash (Linux, macOS, Git Bash, WSL); на Windows используйте Git Bash или WSL, а не PowerShell.

```bash
# 1. установка
pip install -r requirements.txt
# 2. укажите свою LLM
printf 'llm: {url: http://localhost:8000/v1/chat/completions, model: Qwen/Qwen3-8B}\n' > config.yaml
# 3. запуск
python bench.py --config config.yaml --machine my-box
```

`type` по умолчанию — `openai`; `model` — точное имя, под которым сервер отдаёт модель (см. `GET /v1/models`).
Прогон идёт ~5 минут (4 уровня нагрузки × 60 с); `--tests latency` проверяет настройку за ~30 с. Без своих вопросов
берётся встроенный; чтобы мерить на своих, добавьте `data/llm_prompts.jsonl` ([тестовые данные](#тестовые-данные)).

## Что получится

Каждый прогон пишет папку `results/<машина>_<дата>_<время>/` рядом с конфигом. **Сначала откройте `summary.md`**:
вверху — двухстрочный **Answer:** (скорость для одного пользователя и сколько пользователей укладываются в пороги) и
легенда, дальше задержка одного пользователя, уровни нагрузки, качество, пики ресурсов и, если задан `expect:`,
таблица «Expected vs measured». По легенде: p50/p95 — медиана и 95-й перцентиль, уровень нагрузки пройден при ≤1%
неудачных запросов, «N+» — прошёл даже самый высокий проверенный уровень N.

| Файл | Что внутри |
| --- | --- |
| `summary.md` | таблицы выше |
| `results.json` | все числа (p10/p50/p95/среднее) и использованный конфиг |
| `monitor.csv` | раз в секунду: `time, phase, ram_used_gb, gpu_util_pct, gpu_mem_used_mb, power_w, temp_c`; с `--no-monitor` не пишется |
| `llm_answers.jsonl` | ответы из теста качества с результатом `must_contain` (если тест качества шёл с этапом `llm`) |
| `log.txt` | лог консоли |

Что меряется:

- **Задержка, один пользователь:** время до первого токена (TTFT), скорость генерации (ток/с), пауза от конца фразы
  пользователя до первого звука ответа, весь ход — пока не озвучен весь ответ.
- **Нагрузка:** 1, 2, 5, 10... одновременных диалогов, каждый уровень — заданное время; прогон останавливается, когда
  скорость на человека падает ниже порога. Видно, сколько пользователей укладываются в пороги и насколько ASR/TTS
  замедляют LLM.
- **Качество:** WER/CER и скорость ASR относительно реального времени, скорость TTS и разборчивость (синтез →
  распознавание → CER), точность NLU, обязательные слова в ответах LLM (`must_contain`).
- **Ресурсы:** загрузка GPU, видеопамять, мощность, температура и **занятая память системы**; фаза `idle` — исходный
  уровень за `run.idle_s` секунд с уже загруженными моделями. На общей памяти (GB10 / DGX Spark, Apple silicon) ОЗУ
  и есть видеопамять.

Секреты в `results.json`, `log.txt` и `summary.md` скрываются: ключи, в названии которых есть `key`, `token`,
`secret`, `password`, `passwd`, `auth`, `credential`, `bearer` или `headers`, логины и пароли в URL и query-строки URL.

## Весь пайплайн

1. Поднимите модели: LLM и ASR — на любом OpenAI-совместимом сервере, TTS и NLU — на любом HTTP-эндпоинте (или через
   плагины).
2. Заполните в `config.yaml` секции `asr`, `llm`, `tts` и (по желанию) `nlu`.
3. Скопируйте примеры данных в `data/`, русские или английские (для русских задайте `run.chars_per_token: 3`):
   ```bash
   for f in data/*.ru.example.*; do cp "$f" "${f/.ru.example/}"; done                           # русские
   for f in data/*.example.*; do [[ $f == *.ru.example.* ]] || cp "$f" "${f/.example/}"; done   # английские
   ```
   Записей нет? `python make_test_audio.py --config config.yaml` озвучит вопросы вашим TTS в `data/audio/*.wav` и
   `data/asr_manifest.csv` (пути меняются ключами `--out`, `--manifest`). Для замера скорости хватает; WER на
   синтетическом голосе выйдет лучше, чем на живых людях.
4. Запуск: `python bench.py --config config.yaml --machine my-box`.

Ключи: `--tests latency,load,quality` (любой набор; неизвестное имя — ошибка); `--no-monitor` — если модели работают
на другой машине (монитор читает GPU и память локальной). Полный прогон голосового пайплайна на реальных моделях с
уровнями до 10 пользователей — 20–30 минут.
Если WAV-файлов из манифеста нет на диске, голосовые тесты пропускаются, тесты LLM идут.

Чтобы сравнить машины, запустите один конфиг с теми же данными под разными `--machine` (например, `gb10-fp8`,
`rtx5090-fp8`), меняя что-то одно за раз, и сравните файлы `summary.md`.

## Пробный прогон без GPU

`tests/fake_server.py` изображает все этапы на одном порту. Скопируйте и выполните по порядку:

```bash
python tests/fake_server.py 18080 &                                    # оставьте работать
for f in data/*.example.*; do [[ $f == *.ru.example.* ]] || cp "$f" "${f/.example/}"; done
sed 's#localhost:800[0-3]#127.0.0.1:18080#' config.example.yaml > config.yaml
python make_test_audio.py --config config.yaml
python bench.py --config config.yaml --no-monitor --tests latency,quality   # ~1 мин; с нагрузкой ~8 мин
```

Цифры ничего не значат: так проверяют инструмент и смотрят, какие файлы получаются. Эндпоинты — чтобы попробовать
другие типы этапов:

| Эндпоинт | Что изображает | Конфиг этапа |
| --- | --- | --- |
| `/v1/chat/completions` | LLM потоком | `llm: {type: openai}` |
| `/v1/audio/transcriptions` | ASR в стиле OpenAI | `asr: {type: openai}` |
| `/asr` | ASR на JSON, `{"result": {"text": ...}}` | `asr: {type: http, text_field: result.text}` |
| `/predict` | NLU, `{"intent": ...}` | `nlu: {type: http}` |
| `/tts`, `/tts_json`, `/tts_pcm` | TTS: WAV потоком, `{"audio": WAV в base64}`, сырой PCM16 24 кГц | `tts: {type: http}` с `response: wav`, `json_base64` или `pcm16` + `sample_rate: 24000` |

## Конфиг

Каждый ключ прокомментирован в `config.example.yaml`. Нет секции этапа — этап пропускается.

```yaml
llm:
  type: openai                # openai | plugin
  url: http://localhost:8000/v1/chat/completions
  model: Qwen/Qwen3-8B
  max_tokens: 200
  temperature: 0.7
  system_prompt: "You are a helpful voice assistant. Answer in 1-3 sentences."
  extra_body: {}              # добавляется в запрос, например {chat_template_kwargs: {enable_thinking: false}}
asr:
  type: openai                # openai | http | plugin
  url: http://localhost:8001/v1/audio/transcriptions
  model: Qwen/Qwen3-ASR-1.7B
nlu:
  type: http                  # http | plugin
  url: http://localhost:8003/predict
  intent_field: intent        # путь в ответе через точку, например result.intent
tts:
  type: http                  # http | plugin
  url: http://localhost:8002/tts
  response: wav               # wav | json_base64 | pcm16 (нужен sample_rate)
data: {asr_manifest: data/asr_manifest.csv, llm_prompts: data/llm_prompts.jsonl, nlu_set: data/nlu.csv,
       tts_texts: data/tts_texts.txt}
run:
  warmup: 3
  repeats: 20                 # ходов в тесте одного пользователя
  concurrency_levels: [1, 2, 5, 10]
  concurrency_duration_s: 60  # длительность каждого уровня нагрузки
  history_tokens: 4000        # история диалога в каждом запросе
  chars_per_token: 4          # ~4 для английского, ~3 для русского
  stop_below_tps: 1           # остановить нагрузку ниже этой скорости на человека
  request_timeout_s: 300
  idle_s: 5                   # исходный уровень (фаза idle) перед тестами
thresholds:
  comfortable_tps: 20         # скорость на человека, при которой ответ «льётся»
  limit_tps: 10               # минимально допустимая скорость на человека
  pause_max_s: 2.0            # допустимая пауза до ответа (p95)
# expect:                    # необязательно: ваши ожидания, выводятся в «Expected vs measured»
#   single_tps: 25
#   pause_s: "1-2"
#   comfortable_n: "1-2"
#   limit_n: "~10"
#   pipeline_drop_pct: 20     # замедление LLM из-за ASR/TTS, %
```

Форматы встроенных типов:

| Этап | `type` | Запрос | Ответ |
| --- | --- | --- | --- |
| LLM | `openai` | chat completions, `stream: true` | SSE; `usage` читается, если сервер его присылает |
| ASR | `openai` | multipart: `file` + `model` (+ `language`, `extra`) | `{"text": ...}` |
| ASR | `http` | `{audio_field: WAV в base64, ...extra}` | `text_field` (путь через точку) |
| NLU | `http` | `{text_field: текст, ...extra}` | `intent_field` (путь через точку) |
| TTS | `http` | `{text_field: текст, ...extra}` | WAV потоком, `{audio_field: WAV в base64}` или сырой PCM16 |

- У любого встроенного этапа есть `api_key:` (уходит как `Authorization: Bearer ...`) и `headers:`.
- История дополняется репликами до `history_tokens`; каждая начинается со случайного «Session <n>.», чтобы сервер не
  взял готовый префикс из кеша. Текст реплик — `llm.history_filler_user`, `llm.history_filler_assistant`; вопрос по
  умолчанию — `llm.default_prompt`.
- Старый ключ `mode:` работает как `type:` (`mode: json` — это `type: http`).

## Плагины

Всё остальное (gRPC, WebSocket, модель прямо в процессе) подключается небольшим классом на Python:

```yaml
asr:
  type: plugin
  plugin: plugins/my_asr.py:MyASR   # файл относительно конфига или пакет.модуль:Класс
  url: http://localhost:9000/recognize   # остальные ключи секции передаются классу
```

Класс создаётся как `Cls(stage_cfg, cfg, client)` (`client` — общий `httpx.AsyncClient`), может иметь
`async def aclose(self)` и реализует метод своего этапа. За образец удобно брать
[`examples/plugin_echo.py`](../examples/plugin_echo.py) или встроенные адаптеры в `adapters.py`.

| Этап | Метод |
| --- | --- |
| ASR | `async def transcribe(self, wav: bytes) -> str` |
| NLU | `async def classify(self, text: str) -> str \| None` |
| LLM | `def stream(self, messages: list[dict])`: асинхронный генератор пар `(piece, usage)`. `piece` может быть `""` («размышления» или keep-alive); `usage` (`prompt_tokens`, `completion_tokens`) — только во фрагменте, где он пришёл, иначе `None`. Без usage токены считаются по фрагментам стрима. |
| TTS | `def stream(self, text: str)`: асинхронный генератор байтов ответа; `def to_wav(self, raw: bytes) -> bytes`; `streams_audio: bool` (`False` — время до первого звука равно времени всего синтеза) |

LLM-плагин без `model:` подписывается в шапке сводки своим `plugin:`. Свои закрытые плагины держите вне репозитория
или в `plugins_private/` (git его игнорирует).

## Тестовые данные

Файлы в UTF-8; примеры — `data/*.example.*` (английский) и `data/*.ru.example.*` (русский).

| Файл | Формат | Для чего |
| --- | --- | --- |
| `asr_manifest.csv` | CSV `audio,text`: путь к WAV относительно манифеста, эталонный текст | WER/CER; фразы пользователя в тестах пайплайна |
| `llm_prompts.jsonl` | по одному `{"prompt": "...", "must_contain": ["слово", ...]}` в строке | вопросы для LLM, проверка `must_contain` |
| `nlu.csv` | CSV `text,intent` | точность намерений |
| `tts_texts.txt` | по одной фразе в строке | скорость TTS и разборчивость |

Тексты сравниваются в нижнем регистре, без знаков препинания, `ё` → `е`. `must_contain` ищется как подстрока (основа
`отмен` совпадёт и с «отменён», и с «отменить»). Лучше всего — записи живых пользователей, WAV 16 кГц моно; 30–100
файлов дают устойчивый WER, для NLU нужно 20+ фраз на намерение.

## Заметки по железу

- [gb10.md](gb10.md) (на английском): NVIDIA GB10 / DGX Spark — чем отличается, как планировать память, как читать
  результаты.
- [setup/README.md](../setup/README.md) (на английском): установка и запуск любой модели с Hugging Face в vLLM
  (профили `gb10`, `rtx`).

## Тесты

```bash
pip install -r requirements.txt pytest
python -m pytest
```

GPU и модели не нужны: тесты используют фейковый сервер из [пробного прогона](#пробный-прогон-без-gpu).

## Лицензия

MIT, см. [LICENSE](../LICENSE).
