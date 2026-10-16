export type Lang = 'ko' | 'en';

export const TISTORY = 'https://mynews76154.tistory.com';
export const GITHUB = 'https://github.com/tjehdgus';
const t = (id: number) => `${TISTORY}/${id}`;

export const ui = {
  ko: {
    name: '서동현',
    role: 'AI 엔지니어',
    nav: { home: '홈', about: '소개', projects: '프로젝트', posts: '리포트' },
    heroKicker: 'AI Engineer · Agents · RAG · LLM in production',
    heroTitle: ['모델을 고르는 이유와', '검증하는 방법을 기록합니다'],
    heroDesc:
      'RAG와 에이전트 시스템을 만들고, ML 대회에서 검증 설계를 배워 온 AI 엔지니어입니다. 점수보다 왜 그렇게 결정했는지를 남깁니다.',
    ctaProjects: '프로젝트 보기',
    ctaBlog: '티스토리 블로그',
    featured: '대표 프로젝트',
    recent: '기술 리포트',
    allProjects: '전체 프로젝트',
    allPosts: '전체 리포트',
    readOn: '개발 기록',
    code: '코드',
    result: '결과',
    stack: '사용 기술',
    blogNote: '대회 회고와 트렌드 리뷰, 개발 과정 기록은 티스토리 블로그에 더 자주 올립니다.',
    postsIntro: '프로젝트별 기술 리포트입니다. 문제 정의, 데이터, 파이프라인, 실험 결과를 코드 기준으로 정리했고, 모든 글이 영어로도 있습니다.',
    projectsIntro: 'RAG·에이전트, AI 서비스, ML 대회, 실무 프로젝트를 정리했습니다.',
    translated: '이 글의 영어판',
    originalOn: '개발 과정 기록 (티스토리)',
    updated: '수정',
    langSwitch: 'EN',
    themeToggle: '테마 전환',
    groups: { rag: 'RAG · 에이전트', service: 'AI 서비스', comp: 'ML 대회 · 공모전', work: '실무 프로젝트' },
  },
  en: {
    name: 'Donghyun Seo',
    role: 'AI Engineer',
    nav: { home: 'Home', about: 'About', projects: 'Projects', posts: 'Reports' },
    heroKicker: 'AI Engineer · Agents · RAG · LLM in production',
    heroTitle: ['Notes on why a model was chosen', 'and how it was validated'],
    heroDesc:
      'I build RAG and agent systems, and I learned validation design the hard way in ML competitions. I write down why each decision was made, not just the score.',
    ctaProjects: 'View projects',
    ctaBlog: 'Korean blog',
    featured: 'Featured projects',
    recent: 'Technical reports',
    allProjects: 'All projects',
    allPosts: 'All reports',
    readOn: 'Build log (KO)',
    code: 'Code',
    result: 'Result',
    stack: 'Stack',
    blogNote: 'Competition retrospectives, trend reviews and build logs are posted more often on my Korean blog.',
    postsIntro: 'Technical reports for each project: problem, data, pipeline and experiments, written from the code. Every report also exists in Korean.',
    projectsIntro: 'RAG and agents, AI services, ML competitions and work projects.',
    translated: 'Korean version',
    originalOn: 'Build log on my Korean blog',
    updated: 'Updated',
    langSwitch: '한국어',
    themeToggle: 'Toggle theme',
    groups: { rag: 'RAG · Agents', service: 'AI services', comp: 'ML competitions', work: 'Work projects' },
  },
} as const;

export type Project = {
  id: string;
  group: 'rag' | 'service' | 'comp' | 'work';
  featured?: boolean;
  period: string;
  title: Record<Lang, string>;
  summary: Record<Lang, string>;
  result: Record<Lang, string>;
  stack: string[];
  repo?: string;
  posts: string[];
  postKey?: string;
};

export const projects: Project[] = [
  {
    id: 'kftc',
    postKey: 'kftc-rag',
    group: 'rag',
    featured: true,
    period: '2026.04 – 05',
    title: { ko: '공정위 의결서 RAG', en: 'Fair Trade Commission decision RAG' },
    summary: {
      ko: '의결서 500건(31,877청크)에서 근거 청크 5개와 답변을 30초 안에 돌려주는 시스템. 8B 이하 로컬 LLM 제약 아래 하이브리드 검색과 리랭커 2종 RRF, EXAONE 3.5 7.8B LoRA를 붙였다. 2인 팀에서 LLM 학습·생성 평가와 검색 인프라 일부를 맡았다.',
      en: 'Returns five evidence chunks and an answer within 30 seconds over 500 decisions (31,877 chunks), under a local-LLM-only (≤8B) rule. Hybrid retrieval, two rerankers fused with RRF, and a LoRA on EXAONE 3.5 7.8B. In a team of two, I owned LLM training, generation evaluation and part of the retrieval stack.',
    },
    result: {
      ko: '자체 평가 59.91 → 67.76점. 평가셋 누수로 나온 72.96점은 직접 무효 처리',
      en: 'Internal eval 59.91 → 67.76. We voided our own 72.96 after finding eval-set leakage',
    },
    stack: ['BM25', 'BGE-m3-ko', 'RRF', 'Reranker', 'EXAONE 3.5', 'LoRA', 'Contextual Chunking'],
    repo: `${GITHUB}/KFTC`,
    posts: [t(1), t(2), t(3)],
  },
  {
    id: 'clinical',
    postKey: 'clinical-agent',
    group: 'rag',
    featured: true,
    period: '2026.07 – 08',
    title: { ko: '임상 실패 조기경보 에이전트', en: 'Clinical trial failure early-warning agent' },
    summary: {
      ko: '임상 설계 단계에서 중단 위험을 예측하고, 수정안을 다시 예측하고, 규제 가이드라인으로 검토하는 에이전트. LangGraph 서브에이전트 5개에 XGBoost 중단확률 모델, RAG 2종, 검증 노드를 묶었다.',
      en: 'An agent that predicts the risk of a trial being stopped at the design stage, re-scores proposed revisions, and reviews them against regulatory guidelines. Five LangGraph sub-agents, an XGBoost stop-probability model, two RAG pipelines and a verification node.',
    },
    result: {
      ko: '중단확률 AUC 0.730, ECE 0.018, 인용한 NCT 번호 실재율 100%(53건)',
      en: 'Stop-probability AUC 0.730, ECE 0.018, 100% of cited NCT IDs verified real (53)',
    },
    stack: ['LangGraph', 'XGBoost', 'RAG', 'Calibration'],
    posts: [t(4), t(5)],
  },
  {
    id: 'lawkeeper',
    postKey: 'lawkeeper',
    group: 'rag',
    period: '',
    title: { ko: 'LawKeeper 법률 상담 RAG 챗봇', en: 'LawKeeper legal-advice RAG chatbot' },
    summary: {
      ko: '판례·법령·약관·상담 데이터를 ChromaDB 5개 컬렉션에 넣은 멀티턴 RAG. 사건 정보가 없는 후속 질문은 첫 상황 설명과 합쳐 다시 쓴 문장으로 검색하게 했다. 4인 팀으로 웹 배포까지 했다.',
      en: 'Multi-turn RAG over case law, statutes, terms and Q&A in five ChromaDB collections. Follow-up questions that carry no case details are merged with the user’s original situation and rewritten before retrieval. Built and deployed to the web by a team of four.',
    },
    result: {
      ko: '후속 질문에서 검색이 무너지는 문제를 대화 처리 코드와 프롬프트를 열 번 가까이 고쳐 개선',
      en: 'Fixed retrieval breaking on follow-up questions after reworking the dialogue code and prompts about ten times',
    },
    stack: ['ChromaDB', 'ko-sbert', 'GPT-4o-mini', 'FastAPI', 'Cloud Run'],
    repo: `${GITHUB}/lawkeeper_rag_openAI`,
    posts: [t(24), t(25)],
  },
  {
    id: 'diary',
    postKey: 'maeum-diary',
    group: 'service',
    featured: true,
    period: '2026.03 – 07',
    title: { ko: '마음일기 AI 일기 앱', en: 'Maeum-ilgi, an AI diary app' },
    summary: {
      ko: '일기를 쓰면 Claude가 감정을 분석하고, 과거 일기를 기억하는 AI와 대화할 수 있는 1인 개발 서비스. 13개 기능을 4개로 줄이고, 골든셋 394건으로 감정 분석 프롬프트를 고쳤다. pgvector로 과거 일기를 의미 검색한다.',
      en: 'A solo-built service: Claude analyses the emotion of each entry, and you can talk to an AI that remembers past entries. Cut 13 features down to 4 and iterated the emotion prompt on a 394-item golden set. Past entries are retrieved semantically with pgvector.',
    },
    result: {
      ko: '감정 분석 정확도 53.8% → 71.6%, 4개월 185커밋으로 배포',
      en: 'Emotion accuracy 53.8% → 71.6%; shipped after 185 commits over four months',
    },
    stack: ['FastAPI', 'React', 'Claude', 'Supabase', 'pgvector', 'PWA'],
    posts: [t(7), t(8), t(9), t(10)],
  },
  {
    id: 'session-flow',
    postKey: 'session-flow',
    group: 'service',
    featured: true,
    period: '2026.10',
    title: { ko: 'Session Flow', en: 'Session Flow' },
    summary: {
      ko: '여러 Claude Code 세션이 주고받은 메시지와 파일 수정 diff를 IDE 안에서 그래프로 보여주는 개인 도구. hook 기반 플러그인이 로컬 파일에 기록하고, VS Code 확장이 그 파일을 읽는다. 서버가 없다.',
      en: 'A personal tool that draws the messages exchanged between Claude Code sessions, and the file diffs they made, as a graph inside the IDE. A hook-based plugin writes to local files and a VS Code extension reads them. No server.',
    },
    result: {
      ko: '하루 18커밋으로 v0.9.0, 확장 JS 약 2,350줄과 테스트 약 920줄',
      en: 'v0.9.0 in one day across 18 commits; ~2,350 lines of extension JS and ~920 lines of tests',
    },
    stack: ['Claude Code hooks', 'VS Code API', 'Node.js'],
    repo: `${GITHUB}/session-flow`,
    posts: [t(26)],
  },
  {
    id: 'etri',
    postKey: 'etri-sleep',
    group: 'comp',
    featured: true,
    period: '2026.06',
    title: { ko: 'ETRI 수면 지표 예측', en: 'ETRI sleep-quality prediction' },
    summary: {
      ko: '폰·워치 센서 하루치로 그날 밤 수면 7개 지표를 예측했다. 피험자 10명, 학습 450박의 작은 데이터라 검증 설계가 점수를 갈랐다. 9개 소스를 타겟별로 블렌드하고 forward-chaining CV로 재가중했다.',
      en: 'Predicted seven binary sleep metrics for the coming night from one day of phone and watch sensors. With 10 subjects and 450 training nights, validation design decided the score. Nine sources blended per target and re-weighted with forward-chaining CV.',
    },
    result: {
      ko: 'Private 22위(0.57849), ICTC 2026 논문 채택(게재는 포기)',
      en: 'Private rank 22 (0.57849); paper accepted at ICTC 2026 (withdrawn before publication)',
    },
    stack: ['LightGBM', 'PyTorch', 'SigLIP', 'Forward-chaining CV'],
    repo: `${GITHUB}/etri-2026-sleep-prediction`,
    posts: [t(15), t(11), t(14)],
  },
  {
    id: 'wind',
    postKey: 'baram-wind',
    group: 'comp',
    period: '2026.07',
    title: { ko: 'BARAM 2026 풍력발전량 예측', en: 'BARAM 2026 wind power forecasting' },
    summary: {
      ko: '기상 예보만으로 풍력단지 시간별 발전량을 예측했다. 대회 산식 그대로 채점하는 검증 하네스를 먼저 만들고, MOS로 예보 풍속을 보정한 뒤 BiGRU seq2seq를 주력으로 블렌드했다.',
      en: 'Forecast hourly wind-farm output from weather forecasts alone. Built a validation harness that scores with the exact competition metric first, corrected forecast wind speed with MOS, then blended around a BiGRU seq2seq model.',
    },
    result: {
      ko: 'MOS 보정으로 풍속 상관 0.88 → 0.91, 로컬·리더보드 상관 약 0.91 유지',
      en: 'MOS raised wind-speed correlation 0.88 → 0.91; local vs leaderboard correlation held at ~0.91',
    },
    stack: ['BiGRU', 'MOS', 'LightGBM', 'TabPFN v2'],
    posts: [t(12)],
  },
  {
    id: 'car',
    postKey: 'car-classification',
    group: 'comp',
    period: '2025.06',
    title: { ko: '396종 차량 이미지 분류', en: '396-class car image classification' },
    summary: {
      ko: '차량 이미지 한 장으로 396개 차종을 분류했다. 노이즈 라벨을 정제하고 백본 5종을 비교해 ConvNeXt V2를 골랐고, LogLoss 대회라 EMA·SWA·Temperature Scaling으로 확률을 다듬었다.',
      en: 'Classified 396 car models from a single image. Cleaned noisy labels, compared five backbones and picked ConvNeXt V2, then refined probabilities with EMA, SWA and temperature scaling for a LogLoss metric.',
    },
    result: { ko: '748팀 중 36위 (Private 0.12452)', en: 'Rank 36 of 748 teams (private 0.12452)' },
    stack: ['ConvNeXt V2', 'EMA', 'SWA', 'TTA'],
    repo: `${GITHUB}/dacon_car_convnext_v2`,
    posts: [t(16), t(17)],
  },
  {
    id: 'elec',
    postKey: 'power-forecasting',
    group: 'comp',
    period: '2025.08',
    title: { ko: '건물 100개 전력사용량 예측', en: 'Power demand forecasting for 100 buildings' },
    summary: {
      ko: '건물 100개의 7일간 시간별 전력을 예측했다. 냉방도시·불쾌지수·휴무일 같은 도메인 피처를 넣고, 유형별 모델 대신 건물마다 가장 좋은 모델을 골라 가중평균했다.',
      en: 'Forecast seven days of hourly demand for 100 buildings. Added domain features such as cooling degree hours, discomfort index and closing days, and picked the best model per building instead of per building type.',
    },
    result: { ko: '934팀 중 18위', en: 'Rank 18 of 934 teams' },
    stack: ['XGBoost', 'LightGBM', 'CatBoost'],
    repo: `${GITHUB}/dacon_elec_catboost`,
    posts: [t(18), t(19)],
  },
  {
    id: 'kwater',
    postKey: 'kwater-drought',
    group: 'comp',
    period: '2024',
    title: { ko: 'K-water 가뭄 단계 예측', en: 'K-water drought stage prediction' },
    summary: {
      ko: 'SPI6 기반 가뭄 단계를 분류하고 R Shiny 대시보드로 보여줬다. 예측 시점에 실제로 알 수 있는 변수만 남겨 12개를 5개로 줄였다.',
      en: 'Classified SPI6-based drought stages and served them in an R Shiny dashboard. Reduced 12 variables to 5 by keeping only what is actually known at prediction time.',
    },
    result: { ko: '장려상, 자체 테스트 F1 0.933', en: 'Honourable mention; internal test F1 0.933' },
    stack: ['R', 'randomForest', 'Shiny', 'Oracle'],
    repo: `${GITHUB}/K_water_randomforest`,
    posts: [t(20), t(21)],
  },
  {
    id: 'sejong',
    postKey: 'sejong-siting',
    group: 'comp',
    period: '2024',
    title: { ko: '세종시 전기자전거 거치대 입지', en: 'E-bike dock siting for Sejong City' },
    summary: {
      ko: '정답이 없는 입지 선정 문제를 서울 따릉이 대여소를 정답 대용으로 삼아 검증했다. 통행속도로 도로 스트레스 지수를 만들고, 동별로 K-Means 후 MCLP 또는 OPTICS를 골랐다.',
      en: 'Validated a siting problem with no ground truth by using Seoul bike-share stations as a proxy. Built a road-stress index from traffic speeds, then chose MCLP or OPTICS per district after K-Means.',
    },
    result: { ko: '본선 진출', en: 'Finalist' },
    stack: ['K-Means', 'MCLP', 'OPTICS'],
    repo: `${GITHUB}/sejong_kmeans`,
    posts: [t(22), t(23)],
  },
  {
    id: 'work-rag',
    postKey: 'product-rag-chatbot',
    group: 'work',
    period: '',
    title: { ko: '제품 문의 RAG 챗봇', en: 'Product inquiry RAG chatbot' },
    summary: {
      ko: '사내 문서를 근거로 고객 제품 문의에 답하고, 사람이 처리할 요청은 상담 신청으로 넘기는 RAG 챗봇.',
      en: 'A RAG chatbot that answers customer product questions from internal documents and hands off requests that need a human.',
    },
    result: {
      ko: 'ChromaDB · bge-m3 · Presidio PII 마스킹 · WebSocket 스트리밍',
      en: 'ChromaDB · bge-m3 · Presidio PII masking · WebSocket streaming',
    },
    stack: ['bge-m3', 'ChromaDB', 'LangChain', 'FastAPI', 'Azure OpenAI'],
    posts: [t(13)],
  },
  {
    id: 'cost',
    postKey: 'cost-simulator',
    group: 'work',
    period: '2026.01 – 02',
    title: { ko: '원가율 예측 시뮬레이터', en: 'Cost-ratio simulator' },
    summary: {
      ko: '영업 담당자가 견적 전에 원가율을 확인하는 웹 시뮬레이터.',
      en: 'A web simulator that lets sales check the cost ratio before sending a quote.',
    },
    result: {
      ko: 'RandomForest 회귀 · FastAPI · Azure Container Apps',
      en: 'RandomForest regression · FastAPI · Azure Container Apps',
    },
    stack: ['RandomForest', 'FastAPI', 'Azure'],
    posts: [t(6)],
  },
];
