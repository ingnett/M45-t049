// ==UserScript==
// @name         CVAT 3D - M45 Box Difficulty & Priority Ranker
// @namespace    https://github.com/ingnett/M45-t049
// @version      1.0.0
// @description  Xếp hạng và định vị trực tiếp Bounding Box Khó trên giao diện CVAT 3D (Đề tài M45)
// @author       Nhóm M45 (Hưng, Dũng, Đường)
// @match        *://*/*tasks/*
// @match        *://*/*jobs/*
// @match        http://localhost:8080/*
// @match        http://127.0.0.1:8080/*
// @grant        none
// ==/UserScript==

(function () {
  'use strict';

  console.log('[M45 CVAT Plugin] Đang khởi chạy hệ thống Xếp hạng Độ khó Box M45...');

  // Cấu hình ngưỡng chuẩn M45
  const THRESHOLDS = {
    car: { minPts: 8, easyPts: 25, maxDist: 35.0, easyDist: 20.0 },
    pedestrian: { minPts: 4, easyPts: 10, maxDist: 25.0, easyDist: 15.0 },
  };

  function evaluateDifficulty(pts, dist, isCar = true) {
    const cfg = isCar ? THRESHOLDS.car : THRESHOLDS.pedestrian;
    let score = 2.5;
    let tier = 'MEDIUM';
    let color = '#F59E0B';
    let risk = 'Bình thường';

    if (pts < cfg.minPts || dist >= cfg.maxDist) {
      tier = 'HARD';
      score = pts <= 3 || dist >= 45.0 ? 4.8 : 4.2;
      color = '#EF4444';
      risk = '🔴 Lộn Yaw 180°, Lệch tâm, Co cụm kích thước';
    } else if (pts >= cfg.easyPts && dist < cfg.easyDist) {
      tier = 'EASY';
      score = 1.4;
      color = '#10B981';
      risk = '🟢 Đủ điểm, an toàn duyệt nhanh';
    }

    return { score, tier, color, risk };
  }

  // Tạo UI HUD nổi trong CVAT
  function injectM45Hud() {
    if (document.getElementById('m45-cvat-hud')) return;

    const hud = document.createElement('div');
    hud.id = 'm45-cvat-hud';
    hud.style.cssText = `
      position: fixed;
      top: 60px;
      right: 20px;
      width: 320px;
      background: rgba(15, 23, 42, 0.95);
      border: 1px solid #38bdf8;
      box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.6), 0 0 15px rgba(56, 189, 248, 0.2);
      border-radius: 12px;
      color: #f8fafc;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      font-size: 12px;
      z-index: 999999;
      backdrop-filter: blur(8px);
      padding: 14px;
      transition: all 0.2s ease;
    `;

    hud.innerHTML = `
      <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #334155; padding-bottom: 8px; margin-bottom: 10px;">
        <div style="font-weight: 800; font-size: 13px; color: #38bdf8; display: flex; align-items: center; gap: 6px;">
          <span>🎯</span> M45 BOX RANKER
        </div>
        <div style="display: flex; gap: 6px;">
          <button id="m45-min-btn" style="background: none; border: none; color: #94a3b8; cursor: pointer; font-size: 14px;">−</button>
          <button id="m45-close-btn" style="background: none; border: none; color: #94a3b8; cursor: pointer; font-size: 14px;">×</button>
        </div>
      </div>

      <div id="m45-hud-body">
        <div style="display: flex; gap: 6px; margin-bottom: 10px;">
          <button id="m45-btn-sort" style="flex: 1; background: #0284c7; color: white; border: none; border-radius: 6px; padding: 7px 10px; font-weight: 600; cursor: pointer; display: flex; align-items: center; justify-content: center; gap: 4px;">
            🏆 Sắp xếp theo Độ khó
          </button>
          <button id="m45-btn-next-hard" title="Phím tắt: Alt + H" style="background: #dc2626; color: white; border: none; border-radius: 6px; padding: 7px 10px; font-weight: 600; cursor: pointer;">
            ⚡ Next Hard
          </button>
        </div>

        <div style="font-size: 11px; color: #94a3b8; margin-bottom: 6px; font-weight: 600; text-transform: uppercase;">
          Bảng Xếp Hạng Box Khó (Ranking)
        </div>
        <div id="m45-ranking-list" style="max-height: 220px; overflow-y: auto; background: rgba(0,0,0,0.3); border-radius: 6px; padding: 6px; border: 1px solid #1e293b;">
          <div style="color: #64748b; text-align: center; padding: 12px 0;">Đang quét các 3D cuboids trong khung hình...</div>
        </div>

        <div style="margin-top: 10px; font-size: 10px; color: #94a3b8; border-top: 1px solid #334155; padding-top: 6px; display: flex; justify-content: space-between;">
          <span>Phím tắt: <b>Alt + H</b> (Nhảy box khó)</span>
          <span style="color: #38bdf8;">Đề tài M45</span>
        </div>
      </div>
    `;

    document.body.appendChild(hud);

    // Xử lý thu nhỏ / đóng
    const minBtn = document.getElementById('m45-min-btn');
    const hudBody = document.getElementById('m45-hud-body');
    minBtn.addEventListener('click', () => {
      hudBody.style.display = hudBody.style.display === 'none' ? 'block' : 'none';
      minBtn.innerText = hudBody.style.display === 'none' ? '+' : '−';
    });

    document.getElementById('m45-close-btn').addEventListener('click', () => {
      hud.style.display = 'none';
    });

    // Nút Sort và Next Hard
    document.getElementById('m45-btn-sort').addEventListener('click', scanAndRankBoxes);
    document.getElementById('m45-btn-next-hard').addEventListener('click', jumpToNextHardBox);

    // Đăng ký phím tắt Alt + H
    window.addEventListener('keydown', (e) => {
      if (e.altKey && (e.key === 'h' || e.key === 'H')) {
        e.preventDefault();
        jumpToNextHardBox();
      }
    });

    // Quét lần đầu
    setTimeout(scanAndRankBoxes, 1500);
  }

  let currentHardIndex = 0;
  let rankedObjectsCache = [];

  // Quét DOM hoặc CVAT state để trích xuất cuboids và xếp hạng
  function scanAndRankBoxes() {
    const listContainer = document.getElementById('m45-ranking-list');
    if (!listContainer) return;

    // Tìm các object items trong sidebar của CVAT
    const cvatObjects = Array.from(document.querySelectorAll('.cvat-objects-sidebar-states-list-item, [data-state-id]'));

    if (cvatObjects.length === 0) {
      listContainer.innerHTML = `
        <div style="padding: 10px; text-align: center; color: #cbd5e1;">
          Không tìm thấy box nào trong khung hình này.<br>
          <small style="color: #64748b;">(Hãy mở một Job 3D và import annotations.xml)</small>
        </div>
      `;
      return;
    }

    const parsed = [];
    cvatObjects.forEach((el, idx) => {
      const text = el.innerText || '';
      const isCar = text.toLowerCase().includes('car') || text.toLowerCase().includes('vehicle');

      // Đọc thuộc tính nếu có sẵn trong text hoặc data attributes
      let rankMatch = text.match(/#(\d+)/);
      let rank = rankMatch ? parseInt(rankMatch[1], 10) : idx + 1;

      let scoreMatch = text.match(/Score[:\s]+([\d.]+)/i);
      let score = scoreMatch ? parseFloat(scoreMatch[1]) : 2.5;

      let tier = score >= 3.6 ? 'HARD' : (score <= 2.2 ? 'EASY' : 'MEDIUM');
      let medal = rank <= 3 ? '🥇' : (rank <= 10 ? '🥈' : '🥉');

      parsed.push({
        element: el,
        index: idx,
        rank: rank,
        score: score,
        tier: tier,
        medal: medal,
        text: text.slice(0, 40),
        color: score >= 3.6 ? '#EF4444' : (score <= 2.2 ? '#10B981' : '#F59E0B'),
      });
    });

    // Sắp xếp theo score giảm dần
    parsed.sort((a, b) => b.score - a.score);
    parsed.forEach((item, r) => {
      item.rank = r + 1;
      item.medal = (r + 1) <= 3 ? '🥇' : ((r + 1) <= 10 ? '🥈' : '🥉');
    });

    rankedObjectsCache = parsed;

    // Render danh sách Leaderboard
    let html = '';
    parsed.forEach((item) => {
      html += `
        <div class="m45-rank-item" data-idx="${item.index}" style="display: flex; justify-content: space-between; align-items: center; padding: 6px 8px; border-bottom: 1px solid #1e293b; cursor: pointer; border-radius: 4px; transition: background 0.15s ease;">
          <div style="display: flex; align-items: center; gap: 6px;">
            <span>${item.medal}</span>
            <b style="color: ${item.color};">#${item.rank.toString().padStart(2, '0')}</b>
            <span style="color: #cbd5e1; max-width: 140px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">${item.tier === 'HARD' ? '🔴 CỰC KHÓ' : (item.tier === 'EASY' ? '🟢 DỄ' : '🟡 TRUNG BÌNH')}</span>
          </div>
          <span style="font-weight: 700; color: ${item.color}; background: rgba(0,0,0,0.4); padding: 2px 6px; border-radius: 4px;">${item.score.toFixed(1)}</span>
        </div>
      `;
    });

    listContainer.innerHTML = html;

    // Bắt sự kiện click vào từng ca để highlight trong CVAT
    listContainer.querySelectorAll('.m45-rank-item').forEach((row) => {
      row.addEventListener('click', () => {
        const idx = parseInt(row.getAttribute('data-idx'), 10);
        focusCvatObject(idx);
      });
    });

    // Đổi màu viền sidebar tương ứng
    parsed.forEach((item) => {
      if (item.element) {
        item.element.style.borderLeft = `4px solid ${item.color}`;
      }
    });

    console.log(`[M45 CVAT Plugin] Đã xếp hạng thành công ${parsed.length} boxes.`);
  }

  function focusCvatObject(idx) {
    if (rankedObjectsCache[idx] && rankedObjectsCache[idx].element) {
      rankedObjectsCache[idx].element.click();
      rankedObjectsCache[idx].element.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  }

  function jumpToNextHardBox() {
    const hardBoxes = rankedObjectsCache.filter((b) => b.tier === 'HARD');
    if (hardBoxes.length === 0) {
      alert('Không có Box Khó nào trong khung hình này!');
      return;
    }
    const target = hardBoxes[currentHardIndex % hardBoxes.length];
    currentHardIndex++;
    if (target && target.element) {
      target.element.click();
      target.element.scrollIntoView({ behavior: 'smooth', block: 'center' });
      console.log(`[M45 CVAT Plugin] Đã nhảy tới Box Khó hạng #${target.rank} (Score: ${target.score})`);
    }
  }

  // Khởi chạy khi trang tải xong
  window.addEventListener('load', () => {
    setTimeout(injectM45Hud, 1000);
  });

  // Tự động kiểm tra nếu URL thay đổi trong Single Page App
  let lastUrl = location.href;
  new MutationObserver(() => {
    const url = location.href;
    if (url !== lastUrl) {
      lastUrl = url;
      setTimeout(injectM45Hud, 1000);
    }
  }).observe(document, { subtree: true, childList: true });

  // Chạy ngay nếu trang đã sẵn sàng
  if (document.readyState === 'complete' || document.readyState === 'interactive') {
    injectM45Hud();
  }
})();
