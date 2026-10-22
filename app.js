// ========================================================
// VLEARN MASTER APP LOGIC
// Interactive Study Platform Engine
// ========================================================

// Application State
const appState = {
  activeTab: 'overview',
  activeDayIndex: 0,
  
  // Flashcards state
  fcIndex: 0,
  fcFilter: 'all',
  fcMastered: new Set(JSON.parse(localStorage.getItem('vlearn_fc_mastered') || '[]')),
  fcLearning: new Set(JSON.parse(localStorage.getItem('vlearn_fc_learning') || '[]')),
  
  // Exam state
  examActive: false,
  examQuestions: [],
  currentQIndex: 0,
  userAnswers: {},
  timerInterval: null,
  timeRemainingSeconds: 1800, // 30 mins
  
  // User progress
  completedDays: new Set(JSON.parse(localStorage.getItem('vlearn_completed_days') || '[]')),
  examHighScore: parseInt(localStorage.getItem('vlearn_exam_high_score') || '0', 10),
  
  // Theme
  theme: localStorage.getItem('vlearn_theme') || 'dark'
};

// Initialize Application
document.addEventListener('DOMContentLoaded', () => {
  initTheme();
  renderOverviewDays();
  renderTheoryPills();
  renderTheoryDay(appState.activeDayIndex);
  initFlashcards();
  initExamSetup();
  initSearch();
  initNavigation();
  updateStatsDisplay();
});

// ========================================================
// NAVIGATION & TABS
// ========================================================
function initNavigation() {
  const navItems = document.querySelectorAll('.nav-item');
  navItems.forEach(item => {
    item.addEventListener('click', () => {
      const tab = item.getAttribute('data-tab');
      switchTab(tab);
    });
  });
}

function switchTab(tabId) {
  appState.activeTab = tabId;

  // Update Nav buttons
  document.querySelectorAll('.nav-item').forEach(b => {
    b.classList.toggle('active', b.getAttribute('data-tab') === tabId);
  });

  // Update Sections
  document.querySelectorAll('.view-section').forEach(s => {
    s.classList.toggle('active', s.id === `view-${tabId}`);
  });

  // Update Header Titles
  const titleMap = {
    overview: { title: 'Tổng quan khóa học', sub: 'Khóa 4 Phase 1 · Khung chương trình chuẩn bị kiểm tra' },
    theory: { title: 'Lý thuyết 14 Day trọng tâm', sub: 'Kiến thức bóc tách từ nền tảng VLearn VinUni AI Thực Chiến' },
    exam: { title: 'Luyện thi trắc nghiệm', sub: 'Mô phỏng đề thi thực chiến có bấm giờ và phân tích lỗi sai' },
    flashcards: { title: 'Flashcards ôn nhanh', sub: 'Ghi nhớ định nghĩa, công thức và cạm bẫy bài thi' },
    cheatsheet: { title: 'Cheat Sheet & Công thức', sub: 'Bảng tra cứu nhanh các chuẩn dữ liệu 2D/3D và chỉ số đo lường' }
  };

  const meta = titleMap[tabId] || titleMap.overview;
  document.getElementById('page-title').textContent = meta.title;
  document.getElementById('page-subtitle').textContent = meta.sub;

  window.scrollTo({ top: 0, behavior: 'smooth' });
}

// ========================================================
// THEME SWITCHER
// ========================================================
function initTheme() {
  const toggleBtn = document.getElementById('theme-toggle');
  if (appState.theme === 'light') {
    document.documentElement.classList.remove('dark');
    toggleBtn.innerHTML = '<i class="fa-solid fa-sun"></i>';
  } else {
    document.documentElement.classList.add('dark');
    toggleBtn.innerHTML = '<i class="fa-solid fa-moon"></i>';
  }

  toggleBtn.addEventListener('click', () => {
    const isDark = document.documentElement.classList.toggle('dark');
    appState.theme = isDark ? 'dark' : 'light';
    localStorage.setItem('vlearn_theme', appState.theme);
    toggleBtn.innerHTML = isDark ? '<i class="fa-solid fa-moon"></i>' : '<i class="fa-solid fa-sun"></i>';
  });
}

// ========================================================
// VIEW 1: OVERVIEW & ROADMAP
// ========================================================
function renderOverviewDays() {
  const container = document.getElementById('overview-day-cards');
  container.innerHTML = '';

  courseData.days.forEach((day, index) => {
    const isDone = appState.completedDays.has(day.id);
    const card = document.createElement('div');
    card.className = `day-card ${isDone ? 'completed' : ''}`;
    card.innerHTML = `
      <div class="day-card-header">
        <span class="day-badge">Day ${day.dayNumber}</span>
        <span class="day-tag">${day.tag}</span>
      </div>
      <h3 class="day-card-title">${day.title}</h3>
      <p class="day-card-desc">${day.summary}</p>
      <div class="day-card-footer">
        <span><i class="fa-regular fa-file-lines"></i> ${day.slidesCount} bài đọc</span>
        <span>${isDone ? '<i class="fa-solid fa-circle-check text-emerald"></i> Đã học' : '<i class="fa-regular fa-circle"></i> Chưa học'}</span>
      </div>
    `;

    card.addEventListener('click', () => {
      appState.activeDayIndex = index;
      switchTab('theory');
      renderTheoryDay(index);
    });

    container.appendChild(card);
  });
}

// ========================================================
// VIEW 2: THEORY
// ========================================================
function renderTheoryPills() {
  const container = document.getElementById('day-selector-bar');
  container.innerHTML = '';

  courseData.days.forEach((day, idx) => {
    const btn = document.createElement('button');
    btn.className = `day-pill-btn ${idx === appState.activeDayIndex ? 'active' : ''}`;
    btn.textContent = `Day ${day.dayNumber}: ${day.title.split('(')[0].trim()}`;
    btn.addEventListener('click', () => {
      appState.activeDayIndex = idx;
      renderTheoryDay(idx);
    });
    container.appendChild(btn);
  });
}

function renderTheoryDay(dayIndex) {
  const day = courseData.days[dayIndex];
  if (!day) return;

  // Update pills
  document.querySelectorAll('.day-pill-btn').forEach((p, i) => {
    p.classList.toggle('active', i === dayIndex);
  });

  const container = document.getElementById('theory-content-container');
  const isDone = appState.completedDays.has(day.id);

  let sectionsHtml = '';
  day.sections.forEach(sec => {
    let bulletsHtml = '';
    if (sec.bullets && sec.bullets.length > 0) {
      bulletsHtml = '<ul>' + sec.bullets.map(b => `<li>${b}</li>`).join('') + '</ul>';
    }
    sectionsHtml += `
      <div class="theory-section">
        <h3><i class="fa-solid fa-chevron-right"></i> ${sec.title}</h3>
        <p>${sec.content}</p>
        ${bulletsHtml}
      </div>
    `;
  });

  container.innerHTML = `
    <div class="theory-header-meta">
      <span class="day-badge">Day ${day.dayNumber} · ${day.id}</span>
      <span class="day-tag">${day.tag}</span>
      <button class="btn btn-sm ${isDone ? 'btn-success' : 'btn-secondary'}" id="btn-toggle-day-complete" style="margin-left: auto;">
        <i class="fa-solid ${isDone ? 'fa-check-double' : 'fa-check'}"></i>
        ${isDone ? 'Đã hoàn thành Day này' : 'Đánh dấu đã học xong'}
      </button>
    </div>

    <h2 class="theory-title">${day.title}</h2>
    
    <div class="theory-body">
      ${sectionsHtml}
      
      <div class="exam-trap-box">
        <h4><i class="fa-solid fa-triangle-exclamation"></i> Cạm bẫy đề thi hay gài:</h4>
        <p>${day.examTrap}</p>
      </div>
    </div>

    <div style="display: flex; justify-content: space-between; margin-top: 32px; padding-top: 24px; border-top: 1px solid var(--border-subtle);">
      <button class="btn btn-secondary" onclick="navigateTheory(-1)" ${dayIndex === 0 ? 'disabled' : ''}>
        <i class="fa-solid fa-arrow-left"></i> Day trước
      </button>
      <button class="btn btn-primary" onclick="navigateTheory(1)" ${dayIndex === courseData.days.length - 1 ? 'disabled' : ''}>
        Day tiếp theo <i class="fa-solid fa-arrow-right"></i>
      </button>
    </div>
  `;

  document.getElementById('btn-toggle-day-complete').addEventListener('click', () => {
    toggleDayComplete(day.id);
    renderTheoryDay(dayIndex);
  });
}

function navigateTheory(offset) {
  const newIndex = appState.activeDayIndex + offset;
  if (newIndex >= 0 && newIndex < courseData.days.length) {
    appState.activeDayIndex = newIndex;
    renderTheoryDay(newIndex);
    window.scrollTo({ top: 120, behavior: 'smooth' });
  }
}

function toggleDayComplete(dayId) {
  if (appState.completedDays.has(dayId)) {
    appState.completedDays.delete(dayId);
  } else {
    appState.completedDays.add(dayId);
  }
  localStorage.setItem('vlearn_completed_days', JSON.stringify([...appState.completedDays]));
  updateStatsDisplay();
  renderOverviewDays();
}

// ========================================================
// VIEW 3: EXAM SIMULATOR
// ========================================================
function initExamSetup() {
  document.getElementById('btn-start-exam').addEventListener('click', startExam);
  document.getElementById('btn-prev-q').addEventListener('click', () => navigateQuestion(-1));
  document.getElementById('btn-next-q').addEventListener('click', () => navigateQuestion(1));
  document.getElementById('btn-submit-exam').addEventListener('click', confirmSubmitExam);
}

function startExam() {
  const filter = document.getElementById('exam-day-filter').value;
  const countVal = document.getElementById('exam-question-count').value;
  const isTimed = document.getElementById('exam-timer-mode').value === 'timed';

  let pool = [...courseData.questions];

  if (filter === 'part1') {
    pool = pool.filter(q => ['D01', 'D02', 'D03', 'D04', 'D05'].includes(q.day));
  } else if (filter === 'part2') {
    pool = pool.filter(q => ['D06', 'D07', 'D08', 'D09'].includes(q.day));
  } else if (filter === 'part3') {
    pool = pool.filter(q => ['D10', 'D11', 'D12', 'D13', 'D14'].includes(q.day));
  }

  // Shuffle pool
  pool.sort(() => Math.random() - 0.5);

  const totalCount = countVal === 'all' ? pool.length : Math.min(parseInt(countVal, 10), pool.length);
  appState.examQuestions = pool.slice(0, totalCount);
  appState.currentQIndex = 0;
  appState.userAnswers = {};
  appState.examActive = true;

  // Set screens
  document.getElementById('exam-setup-card').classList.add('hidden');
  document.getElementById('exam-result-card').classList.add('hidden');
  document.getElementById('exam-active-card').classList.remove('hidden');

  // Timer
  clearInterval(appState.timerInterval);
  if (isTimed) {
    document.getElementById('exam-timer-display').classList.remove('hidden');
    appState.timeRemainingSeconds = totalCount * 60; // 1 min per question
    updateTimerDisplay();
    appState.timerInterval = setInterval(() => {
      appState.timeRemainingSeconds--;
      updateTimerDisplay();
      if (appState.timeRemainingSeconds <= 0) {
        clearInterval(appState.timerInterval);
        submitExam();
      }
    }, 1000);
  } else {
    document.getElementById('exam-timer-display').classList.add('hidden');
  }

  renderQuestionPills();
  renderCurrentQuestion();
}

function updateTimerDisplay() {
  const mins = Math.floor(appState.timeRemainingSeconds / 60);
  const secs = appState.timeRemainingSeconds % 60;
  const formatted = `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  document.getElementById('exam-time-remaining').textContent = formatted;
}

function renderQuestionPills() {
  const container = document.getElementById('q-pills-container');
  container.innerHTML = '';

  appState.examQuestions.forEach((q, idx) => {
    const pill = document.createElement('button');
    const isAnswered = appState.userAnswers[idx] !== undefined;
    const isActive = idx === appState.currentQIndex;

    pill.className = `q-jump-pill ${isAnswered ? 'answered' : ''} ${isActive ? 'active' : ''}`;
    pill.textContent = idx + 1;
    pill.addEventListener('click', () => {
      appState.currentQIndex = idx;
      renderCurrentQuestion();
    });
    container.appendChild(pill);
  });
}

function renderCurrentQuestion() {
  const q = appState.examQuestions[appState.currentQIndex];
  if (!q) return;

  const total = appState.examQuestions.length;
  document.getElementById('current-q-num').textContent = appState.currentQIndex + 1;
  document.getElementById('total-q-num').textContent = total;

  const progressPercent = ((appState.currentQIndex + 1) / total) * 100;
  document.getElementById('exam-progress-bar').style.width = `${progressPercent}%`;

  const container = document.getElementById('exam-question-box');
  const selectedAns = appState.userAnswers[appState.currentQIndex];

  let answersHtml = '';
  const optLetters = ['A', 'B', 'C', 'D'];
  q.options.forEach((opt, optIdx) => {
    const isSel = selectedAns === optIdx;
    answersHtml += `
      <div class="answer-option ${isSel ? 'selected' : ''}" onclick="selectAnswer(${optIdx})">
        <div class="opt-radio">${optLetters[optIdx]}</div>
        <div class="opt-text">${opt}</div>
      </div>
    `;
  });

  container.innerHTML = `
    <span class="q-badge"><i class="fa-solid fa-tag"></i> ${q.day}</span>
    <h3 class="q-text">Câu ${appState.currentQIndex + 1}: ${q.question}</h3>
    <div class="answers-list">${answersHtml}</div>
  `;

  // Update navigation buttons
  document.getElementById('btn-prev-q').disabled = appState.currentQIndex === 0;
  document.getElementById('btn-next-q').textContent = appState.currentQIndex === total - 1 ? 'Xem lại bài' : 'Câu tiếp theo';

  renderQuestionPills();
}

function selectAnswer(optIdx) {
  appState.userAnswers[appState.currentQIndex] = optIdx;
  renderCurrentQuestion();
}

function navigateQuestion(offset) {
  const newIndex = appState.currentQIndex + offset;
  if (newIndex >= 0 && newIndex < appState.examQuestions.length) {
    appState.currentQIndex = newIndex;
    renderCurrentQuestion();
  }
}

function confirmSubmitExam() {
  const answeredCount = Object.keys(appState.userAnswers).length;
  const total = appState.examQuestions.length;
  const unanswered = total - answeredCount;

  if (unanswered > 0) {
    if (!confirm(`Bạn còn ${unanswered} câu chưa trả lời. Bạn có chắc chắn muốn nộp bài thi ngay không?`)) {
      return;
    }
  }
  submitExam();
}

function submitExam() {
  clearInterval(appState.timerInterval);
  appState.examActive = false;

  let correctCount = 0;
  const total = appState.examQuestions.length;

  appState.examQuestions.forEach((q, idx) => {
    if (appState.userAnswers[idx] === q.correct) {
      correctCount++;
    }
  });

  const scorePercent = Math.round((correctCount / total) * 100);

  // Update High Score
  if (scorePercent > appState.examHighScore) {
    appState.examHighScore = scorePercent;
    localStorage.setItem('vlearn_exam_high_score', scorePercent.toString());
  }

  // Switch to Result View
  document.getElementById('exam-active-card').classList.add('hidden');
  const resultCard = document.getElementById('exam-result-card');
  resultCard.classList.remove('hidden');

  let reviewHtml = '';
  appState.examQuestions.forEach((q, idx) => {
    const userChoice = appState.userAnswers[idx];
    const isCorrect = userChoice === q.correct;
    const optLetters = ['A', 'B', 'C', 'D'];

    reviewHtml += `
      <div class="review-item ${isCorrect ? 'correct' : 'wrong'}">
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
          <span class="q-badge">${q.day}</span>
          <span style="font-weight: 700; color: ${isCorrect ? 'var(--accent-emerald)' : 'var(--accent-rose)'};">
            ${isCorrect ? '<i class="fa-solid fa-check"></i> Đúng' : '<i class="fa-solid fa-xmark"></i> Sai'}
          </span>
        </div>
        <h4 style="font-size: 15px; margin-bottom: 10px;">Câu ${idx + 1}: ${q.question}</h4>
        <p style="font-size: 13.5px; margin-bottom: 4px;">
          Lựa chọn của bạn: <strong>${userChoice !== undefined ? `${optLetters[userChoice]}. ${q.options[userChoice]}` : '<span class="text-rose">Chưa trả lời</span>'}</strong>
        </p>
        ${!isCorrect ? `<p style="font-size: 13.5px; color: var(--accent-emerald); margin-bottom: 8px;">
          Đáp án đúng: <strong>${optLetters[q.correct]}. ${q.options[q.correct]}</strong>
        </p>` : ''}
        <div class="review-explanation">
          <strong><i class="fa-solid fa-lightbulb"></i> Giải thích chi tiết:</strong> ${q.explanation}
        </div>
      </div>
    `;
  });

  resultCard.innerHTML = `
    <div class="score-circle">
      <div class="score-num">${scorePercent}%</div>
      <div class="score-total">${correctCount}/${total} câu đúng</div>
    </div>

    <h2>${scorePercent >= 80 ? '🎉 Xuất sắc! Bạn đã sẵn sàng làm bài kiểm tra!' : (scorePercent >= 60 ? '👍 Khá tốt! Hãy ôn kỹ lại các câu sai bên dưới.' : '⚠️ Cần ôn tập thêm các Day còn yếu!')}</h2>
    <p style="color: var(--text-muted); margin-bottom: 24px;">Điểm cao nhất của bạn: <strong>${appState.examHighScore}%</strong></p>

    <div style="display: flex; gap: 12px; justify-content: center; margin-bottom: 32px;">
      <button class="btn btn-primary" onclick="restartExam()"><i class="fa-solid fa-rotate"></i> Làm lại đề khác</button>
      <button class="btn btn-secondary" onclick="switchTab('theory')"><i class="fa-solid fa-book"></i> Về phần lý thuyết</button>
    </div>

    <h3 style="text-align: left; margin-bottom: 16px; border-bottom: 1px solid var(--border-subtle); padding-bottom: 10px;">
      <i class="fa-solid fa-magnifying-glass-chart"></i> Phân tích chi tiết từng câu:
    </h3>
    <div class="review-answers-list">${reviewHtml}</div>
  `;

  updateStatsDisplay();
  window.scrollTo({ top: 120, behavior: 'smooth' });
}

function restartExam() {
  document.getElementById('exam-result-card').classList.add('hidden');
  document.getElementById('exam-setup-card').classList.remove('hidden');
}

// ========================================================
// VIEW 4: FLASHCARDS (3D FLIP)
// ========================================================
function initFlashcards() {
  const daySelect = document.getElementById('fc-day-select');
  daySelect.innerHTML = '<option value="all">Tất cả chủ đề</option>';

  const uniqueDays = [...new Set(courseData.flashcards.map(f => f.day))];
  uniqueDays.forEach(d => {
    const opt = document.createElement('option');
    opt.value = d;
    opt.textContent = d;
    daySelect.appendChild(opt);
  });

  daySelect.addEventListener('change', (e) => {
    appState.fcFilter = e.target.value;
    appState.fcIndex = 0;
    renderFlashcard();
  });

  const card = document.getElementById('active-flashcard');
  card.addEventListener('click', () => {
    card.classList.toggle('flipped');
  });

  document.getElementById('fc-prev-btn').addEventListener('click', () => navigateFlashcard(-1));
  document.getElementById('fc-next-btn').addEventListener('click', () => navigateFlashcard(1));

  document.getElementById('fc-mark-mastered-btn').addEventListener('click', () => {
    const currentList = getFilteredCards();
    const cur = currentList[appState.fcIndex];
    if (cur) {
      appState.fcMastered.add(cur.id);
      appState.fcLearning.delete(cur.id);
      saveFcState();
      navigateFlashcard(1);
    }
  });

  document.getElementById('fc-mark-review-btn').addEventListener('click', () => {
    const currentList = getFilteredCards();
    const cur = currentList[appState.fcIndex];
    if (cur) {
      appState.fcLearning.add(cur.id);
      appState.fcMastered.delete(cur.id);
      saveFcState();
      navigateFlashcard(1);
    }
  });

  // Space key to flip, arrow keys to navigate
  window.addEventListener('keydown', (e) => {
    if (appState.activeTab === 'flashcards' && !['INPUT', 'SELECT', 'TEXTAREA'].includes(e.target.tagName)) {
      if (e.code === 'Space') {
        e.preventDefault();
        card.classList.toggle('flipped');
      } else if (e.code === 'ArrowRight') {
        navigateFlashcard(1);
      } else if (e.code === 'ArrowLeft') {
        navigateFlashcard(-1);
      }
    }
  });

  renderFlashcard();
}

function getFilteredCards() {
  if (appState.fcFilter === 'all') return courseData.flashcards;
  return courseData.flashcards.filter(f => f.day === appState.fcFilter);
}

function renderFlashcard() {
  const cards = getFilteredCards();
  const cardElem = document.getElementById('active-flashcard');
  cardElem.classList.remove('flipped');

  if (cards.length === 0) {
    document.getElementById('fc-term').textContent = 'Không có thẻ';
    document.getElementById('fc-def').textContent = '';
    return;
  }

  if (appState.fcIndex >= cards.length) appState.fcIndex = 0;
  if (appState.fcIndex < 0) appState.fcIndex = cards.length - 1;

  const cur = cards[appState.fcIndex];
  document.getElementById('fc-badge').textContent = cur.day;
  document.getElementById('fc-term').textContent = cur.term;
  document.getElementById('fc-def').textContent = cur.definition;
  document.getElementById('fc-formula').innerHTML = `<code>${cur.formula}</code>`;

  document.getElementById('fc-current-index').textContent = appState.fcIndex + 1;
  document.getElementById('fc-total-count').textContent = cards.length;

  document.getElementById('fc-mastered-count').textContent = appState.fcMastered.size;
  document.getElementById('fc-learning-count').textContent = appState.fcLearning.size;
}

function navigateFlashcard(offset) {
  const cards = getFilteredCards();
  if (cards.length === 0) return;
  appState.fcIndex = (appState.fcIndex + offset + cards.length) % cards.length;
  renderFlashcard();
}

function saveFcState() {
  localStorage.setItem('vlearn_fc_mastered', JSON.stringify([...appState.fcMastered]));
  localStorage.setItem('vlearn_fc_learning', JSON.stringify([...appState.fcLearning]));
  document.getElementById('fc-mastered-count').textContent = appState.fcMastered.size;
  document.getElementById('fc-learning-count').textContent = appState.fcLearning.size;
}

// ========================================================
// GLOBAL SEARCH
// ========================================================
function initSearch() {
  const searchInput = document.getElementById('global-search');
  searchInput.addEventListener('input', (e) => {
    const q = e.target.value.toLowerCase().trim();
    if (!q) return;

    // Search in daysData
    const matchedDay = courseData.days.findIndex(d => 
      d.title.toLowerCase().includes(q) || 
      d.summary.toLowerCase().includes(q) ||
      d.sections.some(s => s.title.toLowerCase().includes(q) || s.content.toLowerCase().includes(q))
    );

    if (matchedDay !== -1 && appState.activeTab !== 'theory') {
      appState.activeDayIndex = matchedDay;
      switchTab('theory');
      renderTheoryDay(matchedDay);
    }
  });
}

// ========================================================
// STATS & PROGRESS UPDATES
// ========================================================
function updateStatsDisplay() {
  const completedCount = appState.completedDays.size;
  document.getElementById('stat-completed-days').textContent = `${completedCount}/14`;
  document.getElementById('theory-progress-badge').textContent = `${completedCount}/14`;
  document.getElementById('stat-exam-high-score').textContent = `${appState.examHighScore}%`;

  const readiness = document.getElementById('readiness-score');
  if (appState.examHighScore >= 85) {
    readiness.textContent = 'Rất cao (Sẵn sàng 100%)';
    readiness.style.color = 'var(--accent-emerald)';
  } else if (appState.examHighScore >= 65) {
    readiness.textContent = 'Khá vững';
    readiness.style.color = 'var(--accent-blue)';
  } else if (appState.examHighScore > 0) {
    readiness.textContent = 'Cần luyện thêm';
    readiness.style.color = 'var(--accent-amber)';
  } else {
    readiness.textContent = 'Chưa thi thử';
    readiness.style.color = 'var(--text-dim)';
  }
}
