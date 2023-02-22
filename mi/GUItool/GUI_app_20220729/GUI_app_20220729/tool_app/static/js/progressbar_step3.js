/*
Work progressbar in step3
*/

//プログレスバー実行のために常時通信
$(function () {
    const progress_bar = $(".progressbar_step3");
    $(progress_bar).hide();

    function progress_step3()
    {
        //プログレスバー用の通信
        let source = new EventSource("/stream/step3/" + exp_id);

        //100%未満
        source.addEventListener('progress-item-step3', function (event) {
            $(progress_bar).show();
            const dataset = JSON.parse(event.data);
            $('.progress-bar').css('width', dataset.persent + '%').attr('aria-valuenow', dataset.persent);
            $('.progress-bar-label').text(dataset.persent + '%');
            $('#status_step3').text(dataset.status);
            
            //処理中はボタン非活性
            btn_disable();

        }, false);

        //100%で終了、画面更新
        source.addEventListener('last-item-step3', function (event) {
            const dataset = JSON.parse(event.data);
            delete source;
            $('.progress-bar').css('width', '100%').attr('aria-valuenow', 100);
            $('.progress-bar-label').text('100%');
            $('#status_step3').text(dataset.status);
            
            location.reload();
        }, false);

        source.addEventListener("finish-item-step3", function(event) {
            console.log("fin");
            source.close();
        }, false);
    }

    if (status_step3 == "progress")
    {
        progress_step3();
    }
    else
    {
        $('#btn_submit_condition').click(function (event) {
            progress_step3();
        });
    };

});