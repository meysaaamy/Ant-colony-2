# best-of — بهترین کدها استخراج شده از ۶ فایل zip

مقایسه کامل: `بازبینی_کدهای_زیپها.md` (در ریشه ریپازیتوری)

## چrisa این پوشه ساخته شد

در ریپازیتوری ۶ فایل zip وجود داشت که هر کدام یک نسخه مختلف از یک شبیه‌ساز مورچه (ant colony) با Next.js + TypeScript + Canvas 2D بودند. من zipها را استخراج کردم، هر فایل منطقی را در تمام ورژن‌ها مقایسه کردم و بهترین نسخه هر بخش را اینجا کپی کردم (نسخه verbatim با هدر توضیحی منبع).

## ترکیب فایل‌ها در پوشه best-of

| پوشه | فایل | منبع (zip) | چرا برنده است |
|-------|-------|-------------|---------------|
| `lib/` | `types.ts` | fable 5.2 | کامل‌ترین نوع‌ها (BakedChampion, BenchmarkResult, EpisodeConfig/Result, Telemetry) |
| `lib/` | `neat.ts` | fable 5.2 | NeuroEvolution کامل + BAKED_CHAMPION ۹۵ نسل + اعتبارسنجی JSON |
| `lib/` | `ant-soa-engine.ts` | fable 5.2 | SoA ۳۰۰۰ lieu، spatial hash، ۳ политика، réflexها، founder wave |
| `lib/` | `pheromone-grid.ts` | fable 5.2 | CFL clamp، پالمر، mixer با smooth() و ۴ مداخلات رنگی |
| `lib/` | `renderer.ts` | fable 5.2 | رندرر ۱۰ لایه بدون blur، adaptive scaler زیر ۵۰ FPS |
| `lib/` | `prng-noise.ts` | fable 5.2 | Rng (Mulberry32, reseed, gauss), Simplex2D + fbm |
| `lib/` | `presets.ts` | fable 5.2 | split-arena ۵۰/۵۰، لابیرنت، build روی هر مستطیل |
| `lib/` | `headless-sim.ts` | fable 5.2 | **فقط در fable**: HeadlessTrainer chunked، seeds، آبلاسیون |
| `scripts/` | `train.ts` | fable 5.2 | **فقط در fable**: evolução، بنچمارک، bake خودکار |
| `tests/` | `simulation.test.ts` | fable 5.2 | **فقط در fable**: ۴۲ تست vitest |
| | `package.json` | fable 5.2 | tsx + vitest + Next 16.4.2 |
| `components/` | ۷ فایل (ProofOfLearningLab و…) | fable 5.2 | رابط کامل دسکتوب (تنها ProofOfLearningLab در fable) |

## مرجع‌های جایگزین (نسخه‌های با ویژگی‌های_UNIQUE_ دیگر)

| مسیر | منبع | ویژگی_UNIQUE_ که worth است به best-of اضافه شود |
|------|------|--------------------------------------------------|
| `engines/swarm-ant-soa-engine.ts` | swarm-rl-simulation-engine | NEAT درون خود موتور (CSR compileNetwork)، `HyperGenes` (۸ ضریب رفتاری)، hallOfFame، wall-sliding، world ۷۲۰×۱۲۸۰ |
| `refs/prismatic-pheromone-grid.ts` | neuroevolutionary-insect-simulation-engine (1) | پد-stride + wsum، mass-tracking، tone-LUT، `PheromoneParams`، `CatmullRomBrush` état‌دار |
| `refs/mobile-bioslider.tsx` | mobile-bioluminescent-ui-design | BioSlider موبایل (دابل‌تاپ reset، technicalTerm، compact) |

## دستورالعمل assemble نسخه نهایی

به عنوان پایه از **fable_5.2** شروع کنید (تنها ورژنی که تست، آموزش و بنچمارک قابل اجرا دارد):

1. `lib/types.ts` → fable + افزودن `PheromoneParams/Vec2/WallStroke/PresetDef` از prismatic و `HyperGenes/ComparisonMode` از swarm.
2. `lib/neat.ts` → fable + (اختیاری) جایگزینی `Math.tanh` با Padé approximant (`ftanh`) و افزودن ماسک `enabled` از prismatic.
3. `lib/pheromone-grid.ts` → موتور diffusion fable + پد-stride/wsum، mass-tracking و tone-LUT از prismatic.
4. `lib/ant-soa-engine.ts` → fable + افزودن `HyperGenes` و `hallOfFame/learning[]` از swarm و wall-sliding.
5. `lib/prng-noise.ts` → fable + افزودن `clamp`, `lerp`, `XorShift32` از swarm.
6. `lib/presets.ts` → fable (همان‌طور است؛ فرمت `WallStroke` از prismatic در صورت نیاز).
7. `lib/headless-sim.ts`, `scripts/train.ts`, `tests/`, `package.json` → fable را همان‌طور نگه دارید (تنها در fable موجود است).
8. `components/` → fable + ایده `BioSlider` از mobile و دایره D3 از astramax اگر به تجربیات سرور-side نیاز دارید.

## نتیجه‌گیری

**fable 5.2 (Cryo-Obsidian Sanctuary)** بهترین و کامل‌ترین کدبیس است؛ سایر zipها فقط بخش‌های تکمیلی (عملکرد، HyperGenes، رابط موبایل،NEAT ساختاری) دارند. اگر هدف یک شبیه‌ساز مورچه آمادۀ توسعه است، fable را پایه بگیرید و بخش‌های smarter از سایر ورژن‌ها را ماژولار به آن اضافه کنید.
