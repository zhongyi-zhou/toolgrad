// JavaScript to handle the card flipping logic
document.addEventListener('DOMContentLoaded', function () {
    const cards = document.querySelectorAll('#toolgrad-showcase .card');
    cards.forEach(card => {
        const flipButtons = card.querySelectorAll('.flip-btn');
        flipButtons.forEach(btn => {
            btn.addEventListener('click', function () {
                card.classList.toggle('is-flipped');
            });
        });
    });
});