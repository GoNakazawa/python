/*
Work progressbar in step1
*/

//プログレスバー実行のために常時通信
$(function () {
    const progress_bar = $(".progressbar_step1");
    $(progress_bar).hide();

    function progress_step1()
    {
        //プログレスバー用の通信
        let source = new EventSource("/stream/step1/" + exp_id);

        //100%未満
        source.addEventListener('progress-item-step1', function (event) {
            $(progress_bar).show();
            const dataset = JSON.parse(event.data);
            $('.progress-bar').css('width', dataset.persent + '%').attr('aria-valuenow', dataset.persent);
            $('.progress-bar-label').text(dataset.persent + '%');
            $('#status_step1').text(dataset.status);
 
            //処理中はボタン非活性
            btn_disable();

        }, false);

        //100%で終了、画面更新
        source.addEventListener('last-item-step1', function (event) {
            const dataset = JSON.parse(event.data);
            delete source;
            $('.progress-bar').css('width', '100%').attr('aria-valuenow', 100);
            $('.progress-bar-label').text('100%');
            $('#status_step1').text(dataset.status);

            location.reload();
        }, false);

        source.addEventListener("finish-item-step1", function(event) {
            console.log("fin");
            source.close();
        }, false);
    }

    if (status_step1 == "progress")
    {
        progress_step1();
    }
    else
    {
        $('#btn_submit_visualization').click(function (event) {
            progress_step1();
        });
    }
});
