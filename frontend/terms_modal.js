/**
 * First-Time Visit Disclaimer Modal
 * Forces user to agree to terms before accessing the site.
 */

document.addEventListener('DOMContentLoaded', () => {
    checkTermsAcceptance();
});

function checkTermsAcceptance() {
    const TERMS_VERSION = '1.0'; // Increment this to force re-acceptance
    const acceptedVersion = localStorage.getItem('terms_accepted_version');

    if (acceptedVersion !== TERMS_VERSION) {
        showTermsModal();
    }
}

function showTermsModal() {
    // Check if modal already exists
    if (document.getElementById('termsModal')) return;

    // Create Modal HTML
    const modalHtml = `
        <div id="termsModal" class="modal-overlay">
            <div class="modal-content terms-modal">
                <div class="modal-header">
                    <h2>⚠️ Tuyên Bố Miễn Trừ Trách Nhiệm</h2>
                </div>
                <div class="modal-body">
                    <p>Chào mừng bạn đến với <strong>Precious Metals AI</strong>.</p>
                    <p>Trước khi tiếp tục, vui lòng đọc và đồng ý với các điều khoản sau:</p>
                    <ul>
                        <li><strong>Dữ liệu tham khảo:</strong> Mọi dự báo và phân tích trên website này hoàn toàn do AI tạo ra và chỉ mang tính chất tham khảo.</li>
                        <li><strong>Không phải lời khuyên tài chính:</strong> Chúng tôi không cung cấp lời khuyên đầu tư. Quyết định mua/bán thuộc hoàn toàn về bạn.</li>
                        <li><strong>Rủi ro thị trường:</strong> Thị trường vàng/bạc biến động khôn lường. AI có thể dự báo sai. Bạn tự chịu trách nhiệm về mọi rủi ro tài chính.</li>
                    </ul>
                    <p class="terms-note">Bằng việc nhấn "Tôi Đồng Ý", bạn xác nhận đã hiểu và chấp nhận các điều khoản trên.</p>
                </div>
                <div class="modal-footer">
                    <button id="acceptTermsBtn" class="accept-btn">Tôi Đồng Ý & Tiếp Tục</button>
                </div>
            </div>
        </div>
    `;

    // Append to body
    document.body.insertAdjacentHTML('beforeend', modalHtml);

    // Add styles dynamically (if not in CSS)
    // We will rely on styles.css for cleaner code, but add some basics just in case

    // Handle Click
    document.getElementById('acceptTermsBtn').addEventListener('click', () => {
        localStorage.setItem('terms_accepted_version', '1.0');
        const modal = document.getElementById('termsModal');
        modal.classList.add('fade-out');
        setTimeout(() => {
            modal.remove();
        }, 300);
    });
}
