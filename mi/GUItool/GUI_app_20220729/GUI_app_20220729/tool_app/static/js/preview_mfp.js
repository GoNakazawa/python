/*
Submit s3_bucket path, objectives and drop_columns to Flask in step1
*/

//Ajax通信でデータ受け渡し
$(function () {

    function preview_mfp()
    {
        //処理中はボタン非活性
        btn_disable();

        //target
        let radius = $('input[name="radius"]').val();
        let bit_num = 4096
        if (check_bitnum == "short")
        {
            bit_num = $('input[name="bit_num"]').val();
        }

        //json形式へ変換
        var json_data = { "radius": radius, "bit_num": bit_num };

        $.ajax({
            data: JSON.stringify(json_data),
            type: 'POST',
            url: '/read_chemical_data/' + exp_id,
            dataType: 'json',
            contentType: 'application/json'
        }).done(function (data) {
            console.log(data.OK);
        }).always(function () {
            location.reload();
        });
    }

    //submit押下時にFlask側へデータを送信
    $('#btn_submit_mfp').click(function (event) {
        event.preventDefault();
        target = document.getElementById("mfp_block");
        target.innerHTML = "処理は数分で終わります。しばらくお待ちください。";     
        //target_en = document.getElementById("mfp_block_en");
        //target_en.innerHTML = "Wait for a minute...";   
        preview_mfp();
        return false;
    });
})